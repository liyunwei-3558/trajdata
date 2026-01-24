import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from shapely.geometry import Point, LineString, LinearRing, Polygon
from PIL import Image, ImageDraw
import math
import pyproj
from typing import List, Dict, Tuple, Optional, Union
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Polygon

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Polygon as PolygonPatch
import matplotlib.pyplot as plt


import datasets.nuscenes_utils as nutils
# import nuscenes_utils as nutils




class LL2XYProjector:
    """经纬度转UTM坐标投影器"""
    def __init__(self, lat_origin: float, lon_origin: float):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin + 180.) / 6) + 1
        self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat: float, lon: float) -> List[float]:
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]
    


def get_type(element: ET.Element) -> Optional[str]:
    """获取way的类型"""
    for tag in element.findall("tag"):
        if tag.get("k") == "type":
            return tag.get("v")
    return None

def get_subtype(element: ET.Element) -> Optional[str]:
    """获取way的子类型"""
    for tag in element.findall("tag"):
        if tag.get("k") == "subtype":
            return tag.get("v")
    return None

def is_drivable_area(element: ET.Element) -> bool:
    """判断是否为可行驶区域"""
    area_tag = False
    curbstone_tag = False
    for tag in element.findall("tag"):
        if tag.get("k") == "area" and tag.get("v") == "yes":
            area_tag = True
        if tag.get("k") == "type" and tag.get("v") == "curbstone":
            curbstone_tag = True
    return area_tag and curbstone_tag

def get_members_info(element: ET.Element) -> Tuple[Optional[int], List[int]]:
    """获取relation的成员信息"""
    outer = None
    inners = []
    for tag in element.findall("member"):
        if tag.get('type') == "way":
            if tag.get('role') == 'outer':
                outer = int(tag.get('ref'))
            elif tag.get('role') == 'inner':
                inners.append(int(tag.get('ref')))
    return outer, inners

def generate_way(element: ET.Element, point_dict: Dict[int, Point]) -> Union[LineString, LinearRing]:
    """生成way的几何对象"""
    node_list = [int(nd.get('ref')) for nd in element.findall("nd")]
    if node_list[0] == node_list[-1]:
        points = [point_dict[id] for id in node_list[:-1]]
        ring = LinearRing([(p.x, p.y) for p in points])
        return ring
    else:
        points = [point_dict[id] for id in node_list]
        line = LineString([(p.x, p.y) for p in points])
        return line

class SindMapEnv:
    """处理OSM地图数据,实现与NuScenesMapEnv兼容的接口"""
    def __init__(self, 
                 map_data_path: str,
                 bounds: List[float] = [-70, -70, 70, 70],
                 layers: List[str] = ['drivable_area', 'carpark_area', 'road_divider', 'lane_divider'],
                 L: int = 256,
                 W: int = 256,
                 device: str = 'cpu',
                 load_lanegraph: bool = False,
                 lanegraph_res_meters: float = 1.0,
                 pix_per_m: float = 4.0,
                 lat_origin: float = 0.0,
                 lon_origin: float = 0.0):
        """初始化地图环境
        :param map_data_path: OSM地图文件路径
        :param bounds: 地图边界 [low_l, low_w, high_l, high_w]
        :param layers: 需要渲染的图层
        :param L: 图像长度(像素)
        :param W: 图像宽度(像素) 
        :param device: 计算设备
        :param load_lanegraph: 是否加载车道图
        :param lanegraph_res_meters: 车道图分辨率
        :param pix_per_m: 每米像素数
        :param lat_origin: 纬度原点
        :param lon_origin: 经度原点
        """
        # 1. 基本参数初始化
        self.map_path = map_data_path
        self.map_name = os.path.splitext(os.path.basename(map_data_path))[0]
        self.map_list = [self.map_name]         
        self.bounds = bounds
        self.layer_names = layers
        self.L = L
        self.W = W
        self.device = torch.device(device)
        self.pix_per_m = pix_per_m


        
        # 2. 初始化投影器
        self.projector = LL2XYProjector(lat_origin, lon_origin)  # 确保这个类已经定义
        
        # 3. 加载地图数据
        self.load_map()  # 这会初始化 point_dict, way_dict, way_types, polygon_dict
        
        # 4. 定义道路和非道路图层
        self.road_list = ['drivable_area', 'road_segment', 'lane']
        self.num_layers = 0
        road_layers = [lay for lay in self.layer_names if lay in self.road_list]
        self.num_layers = 1 if len(road_layers) > 0 else 0
        nonroad_layers = [lay for lay in self.layer_names if lay not in self.road_list]
        self.num_layers += len(nonroad_layers)

        print("\n=== 图层分类 ===")
        print(f"道路图层: {road_layers}")
        print(f"非道路图层: {nonroad_layers}")
        print(f"总图层数: {self.num_layers}")            
        # 5. 创建图层到通道的映射
        self.layer_map = {}

        print("\n=== 图层通道映射 ===")
        for lay in road_layers:
            self.layer_map[lay] = 0
            print(f"{lay} -> 通道 0")

        lay_idx = 1
        for lay in nonroad_layers:
            self.layer_map[lay] = lay_idx
            print(f"{lay} -> 通道 {lay_idx}")
            lay_idx += 1
        
        # 6. 栅格化准备
        print('Rasterizing maps...')
        m_per_pix = 1.0 / pix_per_m
        self.nusc_raster = []
        self.nusc_dx = []

        map_width = self.bounds[2] - self.bounds[0]
        map_height = self.bounds[3] - self.bounds[1]
        msize = np.array([map_height, map_width])
        cur_msize = msize * pix_per_m
        cur_msize = np.round(cur_msize).astype(np.int32)
        cur_dx = msize / cur_msize
        
        print(f"物理尺寸: {map_width}x{map_height} 米")
        print(f"像素尺寸: {cur_msize[1]}x{cur_msize[0]} 像素")
        print(f"分辨率: {cur_dx} 米/像素")
        

       # 初始化空的车道图字典
        self.lane_graphs = {}

        if load_lanegraph:
            print('Loading lane graphs...')
            self.lane_graphs[self.map_name] = self.load_lanegraph(lanegraph_res_meters)
        

        # 8. 栅格化地图
        map_layers = []
        
        # 处理道路图层（合并到一个通道）
        road_layers = [lay for lay in self.layer_names if lay in self.road_list]
        if len(road_layers) > 0:
            road_img = self.get_map_mask(None, 0.0, road_layers, cur_msize)
            road_img = np.clip(np.sum(road_img, axis=0), 0, 1).reshape((1, cur_msize[0], cur_msize[1])).astype(np.uint8)
            map_layers.append(road_img)
            print(f"道路图层形状: {road_img.shape}")
            print(f"道路图层非零像素数: {np.count_nonzero(road_img)}")
 
        # 处理其他图层（每个一个通道）
        other_layers = [lay for lay in self.layer_names if lay not in self.road_list]
        if len(other_layers) > 0:
            other_img = self.get_map_mask(None, 0.0, other_layers, cur_msize)
            map_layers.append(other_img)
            print(f"非道路图层形状: {other_img.shape}")
            for i, layer in enumerate(other_layers):
                print(f"{layer} 非零像素数: {np.count_nonzero(other_img[i])}")

        # 9. 合并所有图层
        map_img = np.concatenate(map_layers, axis=0)
        print(f"Map shape after rasterization: {map_img.shape}")
        
        # 10. 转换为 tensor 并存储
        self.nusc_raster = torch.from_numpy(map_img).unsqueeze(0).to(device)
        self.nusc_dx = torch.from_numpy(np.array([cur_dx])).to(device)

    

    def load_map(self):
        """加载OSM地图数据"""
        print(f"\n开始加载地图: {self.map_path}")
        try:
            tree = ET.parse(self.map_path)
            root = tree.getroot()
            
            # 处理节点
            print("处理节点...")
            self.point_dict = {}
            for node in root.findall("node"):
                lat = float(node.get('lat'))
                lon = float(node.get('lon'))
                [x, y] = self.projector.latlon2xy(lat, lon)
                x += 45  # 27
                y += 45  # 11
                self.point_dict[int(node.get('id'))] = Point(x, y)
            print(f"完成节点处理, 共 {len(self.point_dict)} 个节点")
            
            # 处理way元素
            print("\n处理way元素...")
            self.way_dict = {}
            self.way_types = {}
            self.polygon_dict = {}
            way_count = 0
            for way in root.findall('way'):
                try:
                    way_id = int(way.get('id'))
                    way_type = get_type(way)
                    is_drivable = is_drivable_area(way)
                    
                    string = generate_way(way, self.point_dict)
                    if isinstance(string, LinearRing):
                        try:
                            self.polygon_dict[way_id] = Polygon(string)
                        except Exception as e:
                            print(f"Warning: Failed to create polygon for way {way_id}: {e}")
                    self.way_dict[way_id] = string
                    self.way_types[way_id] = "drivable_area" if is_drivable else way_type
                    way_count += 1
                    
                except Exception as e:
                    print(f"处理way失败: {way.get('id')}, 错误: {str(e)}")
                    continue
            
            print(f"完成way处理, 共 {way_count} 个way")
            print(f"其中多边形 {len(self.polygon_dict)} 个")
            print("Way类型统计:")
            type_count = {}
            for way_type in self.way_types.values():
                type_count[way_type] = type_count.get(way_type, 0) + 1
            for way_type, count in type_count.items():
                print(f"  {way_type}: {count}")
                    
            # 处理relation元素
            print("\n处理relation元素...")
            relation_count = 0
            for relation in root.findall("relation"):
                if get_type(relation) == 'multipolygon':
                    outer_id, inners_list = get_members_info(relation)
                    if outer_id is not None:
                        outer = self.way_dict.get(outer_id)
                        if outer is not None and isinstance(outer, LinearRing):
                            inners = [self.way_dict[id] for id in inners_list if id in self.way_dict and isinstance(self.way_dict[id], LinearRing)]
                            try:
                                self.polygon_dict[outer_id] = Polygon(outer, inners)
                                relation_count += 1
                            except Exception as e:
                                print(f"Warning: Failed to create multipolygon for relation {relation.get('id')}: {e}")
            print(f"完成relation处理, 处理了 {relation_count} 个多边形关系")

            
                            
        except Exception as e:
            print(f"地图加载失败: {str(e)}")
            raise e
        
        coords = []
        for node in root.findall("node"):
            lat = float(node.get('lat'))
            lon = float(node.get('lon'))
            [x, y] = self.projector.latlon2xy(lat, lon)
            x += 45  # 加上-1.8
            y += 45  # 加上107.5
            coords.append([x, y])
        
        coords = np.array(coords)
        print("\n=== UTM坐标范围（米） ===")
        print(f"X范围: [{coords[:,0].min():.2f}, {coords[:,0].max():.2f}]")
        print(f"Y范围: [{coords[:,1].min():.2f}, {coords[:,1].max():.2f}]")

    def is_drivable_area(element: ET.Element) -> bool:
        """判断是否为可行驶区域"""
        for tag in element.findall("tag"):
            if tag.get("k") == "type" and tag.get("v") == "curbstone":
                return True
        return False
    

    # inter_map0.py 中的正确实现
    def _world_to_pixel(self, coords: np.ndarray) -> np.ndarray:
        """将世界坐标转换为像素坐标
        Args:
            coords: 形状为 (N, 2) 的坐标数组，每行是 [x, y]
        Returns:
            形状为 (N, 2) 的像素坐标数组
        """
        # 1. 计算物理空间到像素空间的转换比例
        meters_to_pixels = min(self.L / (self.bounds[2] - self.bounds[0]),
                            self.W / (self.bounds[3] - self.bounds[1]))
        
        # 2. 将坐标归一化到 [-1, 1] 范围
        x_norm = coords[:, 0] / (self.bounds[2] - self.bounds[0]) * 2
        y_norm = coords[:, 1] / (self.bounds[3] - self.bounds[1]) * 2
        
        # 3. 转换到像素坐标 [0, L-1] 和 [0, W-1]
        px = ((x_norm + 1) * self.L / 2).astype(np.int32)
        py = ((y_norm + 1) * self.W / 2).astype(np.int32)
        
        # 4. 确保在有效范围内
        return np.clip(np.column_stack((px, py)), 0, [self.L-1, self.W-1])


    def _pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """将像素坐标转换为世界坐标
        Args:
            px: x方向像素坐标 [0, 255]
            py: y方向像素坐标 [0, 255]
        Returns:
            (x, y): 世界坐标 [-70, 70]
        """
        # 1. 将像素坐标归一化到 [-1, 1]
        x_norm = (px / self.L) * 2 - 1
        y_norm = (py / self.W) * 2 - 1
        
        # 2. 转换到世界坐标
        x = x_norm * (self.bounds[2] - self.bounds[0]) / 2
        y = y_norm * (self.bounds[3] - self.bounds[1]) / 2
        
        return x, y






    def get_map_mask(self, agent_pos, agent_angle, layer_names, canvas_size):
        """获取地图掩码"""
        map_mask = np.zeros((len(layer_names), canvas_size[0], canvas_size[1]), dtype=np.uint8)
        
        # 计算缩放因子
        # 使用 pix_per_m 来确定正确的缩放
        scale = self.pix_per_m  # 每米4个像素
        
        # 计算偏移，使地图居中
        offset_x = 0                     # +100
        offset_y = canvas_size[0]      #  +50
        
        for i, layer_name in enumerate(layer_names):
            img = Image.new('L', (canvas_size[1], canvas_size[0]), 0)
            draw = ImageDraw.Draw(img)
            
            if layer_name == 'drivable_area':
                for way_id, polygon in self.polygon_dict.items():
                    if self.way_types[way_id] == "drivable_area":
                        # 获取多边形的坐标
                        coords = np.array(polygon.exterior.coords)
                        
                        # 应用缩放和偏移
                        pixels = np.zeros_like(coords)
                        pixels[:, 0] = coords[:, 0] * scale + offset_x
                        pixels[:, 1] = -coords[:, 1] * scale + offset_y  # 注意y轴方向
                        
                        # 绘制多边形
                        draw.polygon(pixels.flatten().tolist(), fill=255)
                        
                        # 处理内环
                        for interior in polygon.interiors:
                            coords = np.array(interior.coords)
                            pixels = np.zeros_like(coords)
                            pixels[:, 0] = coords[:, 0] * scale + offset_x
                            pixels[:, 1] = -coords[:, 1] * scale + offset_y
                            draw.polygon(pixels.flatten().tolist(), fill=0)
                            
            elif layer_name in ['road_divider', 'lane_divider']:
                for way_id, line in self.way_dict.items():
                    if ((layer_name == 'road_divider' and self.way_types[way_id] == 'lll') or
                        (layer_name == 'lane_divider' and self.way_types[way_id] in ['virtual', 'line_thin'])):
                        # 获取线段坐标
                        coords = np.array(line.coords)
                        
                        # 应用缩放和偏移
                        pixels = np.zeros_like(coords)
                        pixels[:, 0] = coords[:, 0] * scale + offset_x
                        pixels[:, 1] = -coords[:, 1] * scale + offset_y
                        
                        # 绘制线段
                        for j in range(len(pixels) - 1):
                            draw.line([pixels[j,0], pixels[j,1], 
                                    pixels[j+1,0], pixels[j+1,1]], 
                                    fill=255, width=2)
            
            map_mask[i] = np.array(img)
        
        return map_mask


    


    def get_map_crop(self, scene_graph, map_idx, bounds=None, L=None, W=None):
        """获取地图裁剪
        Args:
            scene_graph: 场景图，包含位置信息
            map_idx: 地图索引
            bounds: 可选，覆盖默认边界
            L, W: 可选，覆盖默认尺寸
        Returns:
            map_obs: 裁剪后的地图观察
        """
        device = scene_graph.pos.device
        B = scene_graph.pos.size(0)
        NA = scene_graph.pos.size(0)
        bounds = self.bounds if bounds is None else bounds
        L = self.L if L is None else L
        W = self.W if W is None else W

        print("\n=== 开始地图裁剪 ===")
        print(f"场景图节点数: {scene_graph.pos.size(0)}")
        print(f"位置信息:\n{scene_graph.pos}")
        print(f"地图索引: {map_idx}")
        
        # 1. 获取位置信息
        mapixes = map_idx[scene_graph.batch]

        print(f"批次索引: {scene_graph.batch}")
        print(f"地图索引: {mapixes}")

        pos_in = scene_graph.pos

        print("\n=== 位置信息 ===")
        print(f"位置形状: {pos_in.shape}")
        print(f"位置范围: [{pos_in[:,:2].min().item():.2f}, {pos_in[:,:2].max().item():.2f}]")
     

        
        # 3. 处理采样维度
        if len(scene_graph.pos.size()) == 3:
            NS = scene_graph.pos.size(1)
            pos_in = pos_in.reshape(-1, 4)  # [N*NS, 4]
            mapixes = mapixes.unsqueeze(1).expand(-1, NS).reshape(-1)
        
        # 4. 获取地图观察
        map_obs = nutils.get_map_obs(
            self.nusc_raster, 
            self.nusc_dx,
            pos_in,
            mapixes,
            bounds,
            L=L,
            W=W
        ).to(device)

        print("\n=== 裁剪结果 ===")
        print(f"输出形状: {map_obs.shape}")
        
        return map_obs

    def get_map_crop_pos(self, pos, mapixes,
                        bounds=None,
                        L=None,
                        W=None):
        '''
        Render local crops around given global positions (assumed UNNORMALIZED).

        :param pos: batched positions (N x 4) (x,y,hx,hy)
        :param mapixes: the map index of each position (N,)
        :params bounds, L, W: overrides bounds, L, W set in constructor

        :returns map_crop: N x C x H x W
        '''
        device = pos.device
        NA = pos.size(0)

        bounds = self.bounds if bounds is None else bounds
        L = self.L if L is None else L
        W = self.W if W is None else W

        # render by indexing into pre-rasterized binary maps
        map_obs = nutils.get_map_obs(self.nusc_raster, self.nusc_dx, pos,
                                    mapixes, bounds, L=L, W=W).to(device)
        
        print(f"map_obs shape in get_map_crop_pos: {map_obs.shape}")
        
        expected_shape = (NA, self.num_layers, L, W)
        if map_obs.shape != expected_shape:
            raise ValueError(f"Unexpected map_obs shape: {map_obs.shape}, expected: {expected_shape}")
        
        return map_obs.to(device)


    def check_collision(self, positions: torch.Tensor) -> torch.Tensor:
        """检查给定位置是否与地图发生碰撞
        :param positions: 位置张量 (N x 2) 或 (N x 4)
        :return: 碰撞标志张量 (N,)
        """
        if positions.size(-1) > 2:
            positions = positions[..., :2]  # 只使用x,y坐标
            
        collisions = []
        for pos in positions:
            x, y = pos.cpu().numpy()
            px, py = self._world_to_pixel(x, y)
            
            # 检查是否在地图范围内
            if 0 <= px < self.L and 0 <= py < self.W:
                # 检查是否在可行驶区域内
                drivable_idx = self.layer_names.index('drivable_area')
                is_drivable = self.layer_maps[drivable_idx, py, px] > 0
                collisions.append(not is_drivable)
            else:
                collisions.append(True)  # 超出地图范围视为碰撞
                
        return torch.tensor(collisions, dtype=torch.bool, device=positions.device)
    
    def visualize_map(self, save_path: Optional[str] = None):
        """可视化地图
        注意: 使用笛卡尔坐标系显示
        """
        # 获取地图数据
        map_data = self.nusc_raster[0].cpu().numpy()  # [num_layers, H, W]
        num_layers = map_data.shape[0]
        
        # 创建子图
        fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5))
        if num_layers == 1:
            axes = [axes]
                
        # 打印调试信息
        print("\n=== 地图可视化信息 ===")
        print(f"地图数据形状: {map_data.shape}")
        print(f"图层数量: {num_layers}")
        print(f"图层名称: {self.layer_names}")
        
        for idx, (ax, layer_name) in enumerate(zip(axes, self.layer_names)):
            # 显示图像
            ax.imshow(map_data[idx], 
                    extent=[self.bounds[0]+70, self.bounds[2]+70, 
                            self.bounds[1]+70, self.bounds[3]+70], 
                    cmap='gray', 
                    origin='lower')
            
            # 设置坐标轴刻度（每20米一个刻度）
            # 设置坐标轴刻度（从0开始）
            xticks = np.arange(0, 141, 20)  # 0到140
            yticks = np.arange(0, 141, 20)  # 0到140
            
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            
            # 添加网格和中心点标记
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='r', linestyle='-', alpha=0.3)
            ax.plot(0, 0, 'r+', markersize=10, label='Origin')
            
            # 设置标题和标签
            ax.set_title(f'{layer_name}\nSum: {np.sum(map_data[idx])}')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            
            # 保持纵横比
            ax.set_aspect('equal')
                
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def load_lanegraph(self, resolution_meters=1.0, eps=0.01):
        """加载车道图，与NuScenes完全相同的结构，但保持现有的数据处理方式"""
        print("\n开始构建车道图...")
        
        # 1. 保持现有的way筛选逻辑
        valid_ways = []
        for way_id, way in self.way_dict.items():
            way_type = self.way_types[way_id]
            if isinstance(way, LineString) and way_type in ['line_thin', 'virtual']:
                valid_ways.append((way_id, way))
        
        print(f"找到 {len(valid_ways)} 条有效way")
        
        if len(valid_ways) == 0:
            print("警告: 没有找到任何有效way!")
            return {
                'xy': np.zeros((1, 2), dtype=np.float32),
                'edges': np.zeros((1, 5), dtype=np.float32),
                'edgeixes': np.zeros((1, 2), dtype=np.int64),
                'in_edges': [[] for _ in range(1)],   # 添加空的in_edges
                'out_edges': [[] for _ in range(1)]   # 添加空的out_edges
            }
        
        # 2. 保持现有的节点和边的收集逻辑
        xys = []  
        edges = []  
        edgeixes = []  
        
        # 3. 添加in_edges和out_edges的初始化
        in_edges = []   # 每个节点的入边列表
        out_edges = []  # 每个节点的出边列表
        
        # 4. 处理每条way
        for way_id, way in valid_ways:
            coords = np.array(way.coords)
            start_idx = len(xys)
            
            # 添加节点
            for x, y in coords:
                xys.append([x, y])
                in_edges.append([])    # 为新节点添加空的入边列表
                out_edges.append([])   # 为新节点添加空的出边列表
            
            # 添加边和连接关系
            for i in range(len(coords) - 1):
                x0, y0 = coords[i]
                x1, y1 = coords[i+1]
                
                dx = x1 - x0
                dy = y1 - y0
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist > eps:
                    dx_norm = dx / dist
                    dy_norm = dy / dist
                    
                    # 添加边信息
                    edges.append([x0, y0, dx_norm, dy_norm, dist])
                    edgeixes.append([start_idx + i, start_idx + i + 1])
                    
                    # 添加连接关系
                    out_edges[start_idx + i].append(start_idx + i + 1)    # 前向连接
                    in_edges[start_idx + i + 1].append(start_idx + i)     # 后向连接
        
        # 5. 转换为numpy数组
        xys = np.array(xys, dtype=np.float32)
        edges = np.array(edges, dtype=np.float32)
        edgeixes = np.array(edgeixes, dtype=np.int64)
        
        # 6. 保持现有的y坐标翻转逻辑
        map_height = self.bounds[3] - self.bounds[1]
        xys[:,1] = map_height - xys[:,1]
        edges[:,1] = map_height - edges[:,1]
        edges[:,3] *= -1
        
        return {
            'xy': xys,           # [L, 2] 节点坐标
            'edges': edges,      # [M, 5] 边信息 [x0, y0, diff_x, diff_y, dist]
            'edgeixes': edgeixes,# [M, 2] 边的连接关系
            'in_edges': in_edges,  # [L] 每个节点的入边列表
            'out_edges': out_edges # [L] 每个节点的出边列表
        }

    def _remove_duplicates(self, xy, eps):
        """去除距离小于eps的重复点"""
        if len(xy) < 2:
                return xy
            
        kept = [True] * len(xy)
        for i in range(1, len(xy)):
            if np.linalg.norm(xy[i] - xy[i-1]) <= eps:
                kept[i] = False
            
        return xy[kept]

    def _check_duplicates(self, xy, eps):
        """检查是否存在距离小于eps的点"""
        if len(xy) < 2:
            return
        
        diffs = xy[1:] - xy[:-1]
        dists = np.linalg.norm(diffs, axis=1)
        assert np.all(dists > eps), f"存在距离小于{eps}的点"

    def _process_edges(self, xys, out_edges, eps):
            """处理边的信息"""
            edges = []
            edgeixes = []
            ee2ix = {}
            
            for i, outs in enumerate(out_edges):
                x0, y0 = xys[i]
                for e in outs:
                    x1, y1 = xys[e]
                    diff = np.array([x1-x0, y1-y0])
                    dist = np.linalg.norm(diff)
                    if dist <= eps:
                        continue
                    
                    diff /= dist
                    ee2ix[(i,e)] = len(edges)
                    edges.append([x0, y0, diff[0], diff[1], dist])
                    edgeixes.append([i, e])
            
            return np.array(edges), np.array(edgeixes), ee2ix

    def _build_connectivity(self):
        """构建车道连接关系"""
        connectivity = {}
        
        # 遍历所有way，找出相连的车道
        for way in self.ways:
            if way.type not in ['line_thin', 'virtual']:
                continue
                
            connectivity[way.id] = {
                'incoming': [],
                'outgoing': []
            }
            
            # 检查way的起点和终点，找出相连的way
            start_point = way.linestring.coords[0]
            end_point = way.linestring.coords[-1]
            
            for other_way in self.ways:
                if other_way.id == way.id or other_way.type not in ['line_thin', 'virtual']:
                    continue
                    
                other_start = other_way.linestring.coords[0]
                other_end = other_way.linestring.coords[-1]
                
                # 检查连接关系
                if np.linalg.norm(np.array(end_point) - np.array(other_start)) < 0.1:
                    connectivity[way.id]['outgoing'].append(other_way.id)
                if np.linalg.norm(np.array(start_point) - np.array(other_end)) < 0.1:
                    connectivity[way.id]['incoming'].append(other_way.id)
        
        return connectivity

    def _get_outgoing_ways(self, way_id):
        """获取从指定way出发的连接way列表"""
        outgoing = []
        if way_id in self.way_dict:
            way = self.way_dict[way_id]
            if isinstance(way, LineString):
                end_point = np.array(way.coords[-1])
                for other_id, other_way in self.way_dict.items():
                    if other_id != way_id and isinstance(other_way, LineString):
                        other_start = np.array(other_way.coords[0])
                        if np.linalg.norm(end_point - other_start) <= 1e-6:
                            outgoing.append(other_id)
        return outgoing

    def _get_incoming_ways(self, way_id):
        """获取连接到指定way的way列表"""
        incoming = []
        if way_id in self.way_dict:
            way = self.way_dict[way_id]
            if isinstance(way, LineString):
                start_point = np.array(way.coords[0])
                for other_id, other_way in self.way_dict.items():
                    if other_id != way_id and isinstance(other_way, LineString):
                        other_end = np.array(other_way.coords[-1])
                        if np.linalg.norm(start_point - other_end) <= 1e-6:
                            incoming.append(other_id)
        return incoming
    
 

    def objs2crop(self, center, obj_center, obj_lw, map_idx, bounds=None, L=None, W=None):
        '''
        converts given objects N x 4 to the crop frame defined by the given center (x,y,hx,hy)
        '''
        bounds = self.bounds if bounds is None else bounds
        L = self.L if L is None else L
        W = self.W if W is None else W
        local_objs = nutils.objects2frame(obj_center.cpu().numpy()[np.newaxis, :, :],
                                          center.cpu().numpy())[0]
        # [low_l, low_w, high_l, high_w]
        local_objs[:, 0] -= bounds[0]
        local_objs[:, 1] -= bounds[1]

        # convert to pix space
        pix2m_L = L / float(bounds[2] - bounds[0])
        pix2m_W = W / float(bounds[3] - bounds[1])
        local_objs[:, 0] *= pix2m_L
        local_objs[:, 1] *= pix2m_W
        pix_objl = obj_lw[:, 0]*pix2m_L
        pix_objw = obj_lw[:, 1]*pix2m_W
        pix_objlw = torch.stack([pix_objl, pix_objw], dim=1)
        local_objs = torch.from_numpy(local_objs)

        return local_objs, pix_objlw
    

    def _rasterize_road_layers(self, layers, size):
        """栅格化道路相关图层"""
        H, W = size
        road_img = np.zeros((len(layers), H, W), dtype=np.uint8)
        
        for i, layer in enumerate(layers):
            if layer == 'drivable_area':
                for way_id, polygon in self.polygon_dict.items():
                    if self.way_types[way_id] == "drivable_area":
                        self._render_polygon_to_array(road_img[i], polygon)
        
        return road_img

    def _rasterize_other_layers(self, layers, size):
        """栅格化非道路图层"""
        H, W = size
        other_img = np.zeros((len(layers), H, W), dtype=np.uint8)
        
        for i, layer in enumerate(layers):
            if layer == 'road_divider':
                for way_id, way in self.way_dict.items():
                    if self.way_types[way_id] in ['lll']:
                        if isinstance(way, LineString):
                            self._render_line_to_array(other_img[i], way)
            elif layer == 'lane_divider':
                for way_id, way in self.way_dict.items():
                    if self.way_types[way_id] in ['virtual', 'line_thin', 'stop_line']:
                        if isinstance(way, LineString):
                            self._render_line_to_array(other_img[i], way)
            elif layer == 'carpark_area':
                for way_id, way in self.way_dict.items():
                    if self.way_types[way_id] == 'parking':
                        if isinstance(way, LinearRing):
                            try:
                                polygon = Polygon(way)
                                self._render_polygon_to_array(other_img[i], polygon)
                            except Exception as e:
                                print(f"Warning: Failed to render parking polygon {way_id}: {e}")
        
        return other_img

    def _render_polygon_to_array(self, array, polygon):
        """将多边形渲染到numpy数组
        Args:
            array: 目标numpy数组
            polygon: shapely Polygon对象
        """
        try:
            # 获取多边形的外环坐标
            coords = np.array(polygon.exterior.coords)
            if len(coords) < 3:  # 确保至少有3个点形成多边形
                print(f"警告: 多边形点数过少 ({len(coords)})")
                return
            
            # 转换为像素坐标
            pixel_coords = []
            for x, y in coords:
                px, py = self._world_to_pixel(x, y)
                pixel_coords.append([px, py])
            pixel_coords = np.array(pixel_coords)
            
            # 创建PIL Image进行绘制
            img = Image.fromarray(array)
            draw = ImageDraw.Draw(img)
            
            # 绘制填充多边形
            draw.polygon(pixel_coords.flatten().tolist(), fill=255)
            
            # 转换回numpy数组
            array[:] = np.array(img)
            
            print(f"    成功渲染多边形: {len(coords)}个点")
            
        except Exception as e:
            print(f"    多边形渲染失败: {str(e)}")
            print(f"    多边形详情: {polygon}")

    def _render_line_to_array(self, array, line):
        """将线段渲染到numpy数组
        Args:
            array: 目标numpy数组
            line: shapely LineString对象
        """
        try:
            # 获取线段坐标
            coords = np.array(line.coords)
            if len(coords) < 2:  # 确保至少有2个点
                print(f"警告: 线段点数过少 ({len(coords)})")
                return
            
            # 转换为像素坐标
            pixel_coords = []
            for x, y in coords:
                px, py = self._world_to_pixel(x, y)
                pixel_coords.append([px, py])
            pixel_coords = np.array(pixel_coords)
            
            # 创建PIL Image进行绘制
            img = Image.fromarray(array)
            draw = ImageDraw.Draw(img)
            
            # 绘制线段
            for i in range(len(pixel_coords)-1):
                p1 = tuple(pixel_coords[i])
                p2 = tuple(pixel_coords[i+1])
                draw.line([p1[0], p1[1], p2[0], p2[1]], fill=255, width=2)
            
            # 转换回numpy数组
            array[:] = np.array(img)
            
        except Exception as e:
            print(f"    线段渲染失败: {str(e)}")
            print(f"    线段详情: {line}")

    def debug_map_crop(self, map_obs: torch.Tensor, positions: torch.Tensor):
        """可视化地图裁剪结果
        Args:
            map_obs: 裁剪后的地图 [N, C, H, W]
            positions: 位置信息 [N, 4]
        """
        plt.figure(figsize=(15, 5))
        
        # 1. 显示原始地图
        plt.subplot(131)
        plt.imshow(self.nusc_raster[0, 0].cpu(), origin='lower')
        plt.plot(positions[:, 0].cpu(), positions[:, 1].cpu(), 'r.')
        plt.title("Full Map")
        
        # 2. 显示第一个裁剪
        plt.subplot(132)
        plt.imshow(map_obs[0, 0].cpu(), origin='lower')
        plt.title("First Crop (Channel 0)")
        
        # 3. 显示所有通道叠加
        plt.subplot(133)
        plt.imshow(torch.sum(map_obs[0], dim=0).cpu(), origin='lower')
        plt.title("All Channels")
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # 使用示例
    map_env = SindMapEnv(
        map_data_path="map_relink_law_save.osm",
        bounds=[-70, -70, 70, 70],
        L=256,
        W=256,
        layers=['drivable_area', 'carpark_area', 'road_divider', 'lane_divider'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        load_lanegraph=True,
        pix_per_m=4.0
    )


    # 检查车道图是否正确加载
    print("\n车道图验证:")
    print(f"可用地图: {list(map_env.lane_graphs.keys())}")
    for map_name, graph in map_env.lane_graphs.items():
        print(f"\n地图 {map_name}:")
        print(f"  节点 (xy): {graph['xy'].shape if graph is not None else None}")
        print(f"  边 (edges): {graph['edges'].shape if graph is not None else None}")
        
        # 打印一些示例数据
        if graph is not None:
            print("\n示例数据:")
            if len(graph['xy']) > 0:
                print(f"第一个节点坐标: {graph['xy'][0]}")
            if len(graph['edges']) > 0:
                print(f"第一条边: {graph['edges'][0]}")
                print("边的格式: [x0, y0, dx, dy, dist]")

    # 打印地图信息
    print("\n地图信息:")
    print(f"点的数量: {len(map_env.point_dict)}")
    print(f"路段数量: {len(map_env.way_dict)}")
    print(f"多边形数量: {len(map_env.polygon_dict)}")


    # 打印地图坐标范围
    print("\n地图坐标范围:")
    if len(map_env.point_dict) > 0:
        all_points = np.array([[p.x, p.y] for p in map_env.point_dict.values()])
        x_coords = all_points[:, 0]
        y_coords = all_points[:, 1]
        
        print(f"X范围: [{x_coords.min():.2f}, {x_coords.max():.2f}]")
        print(f"Y范围: [{y_coords.min():.2f}, {y_coords.max():.2f}]")
        print(f"中心点: [{x_coords.mean():.2f}, {y_coords.mean():.2f}]")

    # 打印路段的坐标范围
    print("\n路段坐标范围:")
    if len(map_env.way_dict) > 0:
        way_points = []
        for way in map_env.way_dict.values():
            # 从 LineString 对象中获取坐标
            coords = np.array(way.coords)
            way_points.extend(coords)
        way_points = np.array(way_points)
        
        print(f"路段X范围: [{way_points[:, 0].min():.2f}, {way_points[:, 0].max():.2f}]")
        print(f"路段Y范围: [{way_points[:, 1].min():.2f}, {way_points[:, 1].max():.2f}]")

    # 打印多边形的坐标范围
    print("\n多边形坐标范围:")
    if len(map_env.polygon_dict) > 0:
        poly_points = []
        for poly in map_env.polygon_dict.values():
            # 从 Polygon 对象中获取坐标
            coords = np.array(poly.exterior.coords)
            poly_points.extend(coords)
        poly_points = np.array(poly_points)
        
        print(f"多边形X范围: [{poly_points[:, 0].min():.2f}, {poly_points[:, 0].max():.2f}]")
        print(f"多边形Y范围: [{poly_points[:, 1].min():.2f}, {poly_points[:, 1].max():.2f}]")
    
 
    # 可视化地图
    map_env.visualize_map()
# 使用示例
