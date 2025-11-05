# -*- coding: utf-8 -*-

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import hashlib
from collections import defaultdict
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
import os


# ==================== Schema设计 ====================

class RobotNavigationSchema:
    """机器人导航知识图谱Schema设计"""

    def __init__(self):
        # 实体类型定义
        self.entity_types = {
            "Algorithm": {
                "description": "导航和路径规划算法",
                "properties": ["name", "description", "time_complexity", "space_complexity", "optimality"],
                "required": ["name", "description"],
                "examples": ["A_star", "Dijkstra", "RRT", "DWA"]
            },
            "Sensor": {
                "description": "机器人感知传感器",
                "properties": ["name", "description", "range", "accuracy", "cost"],
                "required": ["name", "description"],
                "examples": ["Lidar", "Camera", "IMU", "GPS"]
            },
            "Environment": {
                "description": "机器人运行环境类型",
                "properties": ["name", "description", "complexity", "dynamics", "visibility"],
                "required": ["name", "description"],
                "examples": ["Indoor", "Outdoor", "Dynamic", "Urban"]
            },
            "Task": {
                "description": "机器人导航任务",
                "properties": ["name", "description", "difficulty", "duration"],
                "required": ["name", "description"],
                "examples": ["Path_Planning", "Obstacle_Avoidance", "Localization"]
            },
            "Evaluation": {
                "description": "算法性能评价指标",
                "properties": ["name", "description", "unit", "optimization_direction"],
                "required": ["name", "description"],
                "examples": ["Path_Length", "Success_Rate", "Computation_Time"]
            },
            "Platform": {
                "description": "机器人平台类型",
                "properties": ["name", "description", "mobility", "payload"],
                "required": ["name", "description"],
                "examples": ["Wheeled_Robot", "Aerial_Drone", "Humanoid"]
            }
        }

        # 关系类型定义
        self.relation_types = {
            "used_in": {
                "domain": "Algorithm",
                "range": "Environment",
                "description": "算法应用于特定环境",
                "cardinality": "many-to-many",
                "inverse": "supports_algorithm"
            },
            "requires": {
                "domain": ["Algorithm", "Task"],
                "range": "Sensor",
                "description": "算法或任务需要特定传感器",
                "cardinality": "many-to-many"
            },
            "evaluated_by": {
                "domain": "Algorithm",
                "range": "Evaluation",
                "description": "算法用特定指标评价",
                "cardinality": "many-to-many"
            },
            "performs": {
                "domain": "Algorithm",
                "range": "Task",
                "description": "算法执行特定任务",
                "cardinality": "many-to-many"
            },
            "compared_with": {
                "domain": "Algorithm",
                "range": "Algorithm",
                "description": "算法之间的对比关系",
                "cardinality": "many-to-many",
                "symmetric": True
            }
        }

        # 属性约束
        self.property_constraints = {
            "functional_properties": ["has_time_complexity", "has_optimality"],
            "inverse_functional_properties": ["has_name"],
            "transitive_properties": ["subclass_of"],
            "symmetric_properties": ["compared_with"]
        }

    def validate_entity(self, entity_name, entity_type, properties):
        """验证实体是否符合Schema"""
        if entity_type not in self.entity_types:
            return False, f"未知的实体类型: {entity_type}"

        schema = self.entity_types[entity_type]

        # 检查必需属性
        for required_prop in schema["required"]:
            if required_prop not in properties:
                return False, f"实体 {entity_name} 缺少必需属性: {required_prop}"

        return True, "验证通过"

    def validate_relation(self, subject_type, relation, object_type):
        """验证关系是否符合Schema"""
        if relation not in self.relation_types:
            return False, f"未知的关系类型: {relation}"

        schema = self.relation_types[relation]

        # 检查定义域约束
        domain = schema["domain"]
        if isinstance(domain, list):
            if subject_type not in domain:
                return False, f"关系 {relation} 的定义域应为 {domain}, 但主语类型是 {subject_type}"
        else:
            if subject_type != domain:
                return False, f"关系 {relation} 的定义域应为 {domain}, 但主语类型是 {subject_type}"

        # 检查值域约束
        range_ = schema["range"]
        if isinstance(range_, list):
            if object_type not in range_:
                return False, f"关系 {relation} 的值域应为 {range_}, 但宾语类型是 {object_type}"
        else:
            if object_type != range_:
                return False, f"关系 {relation} 的值域应为 {range_}, 但宾语类型是 {object_type}"

        return True, "验证通过"

    def get_schema_statistics(self):
        """获取Schema统计信息"""
        stats = {
            "entity_types": len(self.entity_types),
            "relation_types": len(self.relation_types),
            "entity_details": {},
            "relation_details": {}
        }

        for entity_type, info in self.entity_types.items():
            stats["entity_details"][entity_type] = {
                "description": info["description"],
                "properties_count": len(info["properties"]),
                "required_count": len(info["required"]),
                "examples": info["examples"]
            }

        for relation, info in self.relation_types.items():
            stats["relation_details"][relation] = {
                "description": info["description"],
                "domain": info["domain"],
                "range": info["range"],
                "cardinality": info["cardinality"]
            }

        return stats


# ==================== 知识图谱类 ====================

class RobotNavigationKnowledgeGraph:
    """机器人导航知识图谱"""

    def __init__(self, enable_schema_validation=True):
        self.entities = {}
        self.triples = []
        self.entity_embeddings = {}
        self.graph = None
        self.schema = RobotNavigationSchema()
        self.enable_schema_validation = enable_schema_validation

        # 统计信息
        self.stats = {
            "entities_by_type": defaultdict(int),
            "relations_by_type": defaultdict(int),
            "validation_errors": []
        }

    def define_entities(self):
        """定义机器人导航领域的实体（50-100个）"""
        print("Defining entities...")

        # 算法类实体 (20个)
        algorithm_entities = {
            "A_star": {"type": "Algorithm", "desc": "A star heuristic search path planning algorithm"},
            "Dijkstra": {"type": "Algorithm", "desc": "Dijkstra shortest path algorithm"},
            "RRT": {"type": "Algorithm", "desc": "Rapidly-exploring Random Tree motion planning"},
            "RRT_star": {"type": "Algorithm", "desc": "Optimized RRT converges to optimal solution"},
            "DWA": {"type": "Algorithm", "desc": "Dynamic Window Approach obstacle avoidance"},
            "Potential_Field": {"type": "Algorithm", "desc": "Potential field navigation method"},
            "SLAM": {"type": "Algorithm", "desc": "Simultaneous Localization and Mapping"},
            "AMCL": {"type": "Algorithm", "desc": "Adaptive Monte Carlo Localization"},
            "EKF_SLAM": {"type": "Algorithm", "desc": "Extended Kalman Filter SLAM"},
            "FastSLAM": {"type": "Algorithm", "desc": "Particle filter based SLAM"},
            "PRM": {"type": "Algorithm", "desc": "Probabilistic Road Map planning"},
            "PRM_star": {"type": "Algorithm", "desc": "Optimal probabilistic road map"},
            "FMT": {"type": "Algorithm", "desc": "Fast Marching Tree algorithm"},
            "BIT_star": {"type": "Algorithm", "desc": "Batch Informed Trees star algorithm"},
            "Theta_star": {"type": "Algorithm", "desc": "Any-angle path planning algorithm"},
            "Lazy_PRM": {"type": "Algorithm", "desc": "Lazy probabilistic road map"},
            "SBP": {"type": "Algorithm", "desc": "Search-based planning algorithm"},
            "D_star": {"type": "Algorithm", "desc": "Dynamic A star for dynamic environments"},
            "LPA_star": {"type": "Algorithm", "desc": "Lifelong Planning A star"},
            "AD_star": {"type": "Algorithm", "desc": "Anytime Dynamic A star algorithm"}
        }

        # 传感器类实体 (15个)
        sensor_entities = {
            "Lidar": {"type": "Sensor", "desc": "Laser Detection and Ranging sensor"},
            "Camera": {"type": "Sensor", "desc": "Visual perception camera"},
            "IMU": {"type": "Sensor", "desc": "Inertial Measurement Unit"},
            "GPS": {"type": "Sensor", "desc": "Global Positioning System"},
            "Odometry": {"type": "Sensor", "desc": "Wheel odometry sensor"},
            "Ultrasonic": {"type": "Sensor", "desc": "Ultrasonic distance sensor"},
            "Infrared": {"type": "Sensor", "desc": "Infrared proximity sensor"},
            "RGB_D_Camera": {"type": "Sensor", "desc": "RGB-D depth sensing camera"},
            "Stereo_Camera": {"type": "Sensor", "desc": "Stereo vision camera"},
            "ToF_Camera": {"type": "Sensor", "desc": "Time-of-Flight camera"},
            "Radar": {"type": "Sensor", "desc": "Radio detection and ranging"},
            "Sonar": {"type": "Sensor", "desc": "Sound navigation and ranging"},
            "Compass": {"type": "Sensor", "desc": "Digital compass for orientation"},
            "Encoder": {"type": "Sensor", "desc": "Rotary encoder for position"},
            "Force_Torque": {"type": "Sensor", "desc": "Force and torque sensor"}
        }

        # 环境类实体 (12个)
        environment_entities = {
            "Indoor": {"type": "Environment", "desc": "Structured indoor environments"},
            "Outdoor": {"type": "Environment", "desc": "Unstructured outdoor environments"},
            "Urban": {"type": "Environment", "desc": "Urban environments with buildings"},
            "Dynamic": {"type": "Environment", "desc": "Dynamic environments with moving obstacles"},
            "Cluttered": {"type": "Environment", "desc": "Cluttered environments with obstacles"},
            "Unknown": {"type": "Environment", "desc": "Unknown environments with no prior map"},
            "Structured": {"type": "Environment", "desc": "Well-structured predictable environments"},
            "Unstructured": {"type": "Environment", "desc": "Unpredictable natural environments"},
            "Aerial": {"type": "Environment", "desc": "Aerial navigation environments"},
            "Underwater": {"type": "Environment", "desc": "Underwater navigation environments"},
            "Planetary": {"type": "Environment", "desc": "Planetary exploration environments"},
            "Industrial": {"type": "Environment", "desc": "Industrial automation environments"}
        }

        # 任务类实体 (10个)
        task_entities = {
            "Path_Planning": {"type": "Task", "desc": "Finding optimal path planning"},
            "Obstacle_Avoidance": {"type": "Task", "desc": "Avoiding collisions with obstacles"},
            "Localization": {"type": "Task", "desc": "Estimating robot position"},
            "Mapping": {"type": "Task", "desc": "Building environment map"},
            "Navigation": {"type": "Task", "desc": "Complete navigation system"},
            "Exploration": {"type": "Task", "desc": "Autonomous environment exploration"},
            "Tracking": {"type": "Task", "desc": "Object or path tracking"},
            "Formation": {"type": "Task", "desc": "Multi-robot formation control"},
            "Manipulation": {"type": "Task", "desc": "Object manipulation and grasping"},
            "Inspection": {"type": "Task", "desc": "Environment or structure inspection"}
        }

        # 评价指标类实体 (8个)
        evaluation_entities = {
            "Path_Length": {"type": "Evaluation", "desc": "Total path length metric"},
            "Success_Rate": {"type": "Evaluation", "desc": "Success rate percentage"},
            "Computation_Time": {"type": "Evaluation", "desc": "Computation time required"},
            "Smoothness": {"type": "Evaluation", "desc": "Path smoothness metric"},
            "Safety_Margin": {"type": "Evaluation", "desc": "Safety distance from obstacles"},
            "Energy_Consumption": {"type": "Evaluation", "desc": "Energy consumption metric"},
            "Accuracy": {"type": "Evaluation", "desc": "Navigation accuracy"},
            "Robustness": {"type": "Evaluation", "desc": "System robustness to disturbances"}
        }

        # 平台类实体 (5个)
        platform_entities = {
            "Wheeled_Robot": {"type": "Platform", "desc": "Wheeled mobile robot"},
            "Aerial_Drone": {"type": "Platform", "desc": "Aerial unmanned vehicle"},
            "Humanoid": {"type": "Platform", "desc": "Humanoid robot"},
            "Tracked_Robot": {"type": "Platform", "desc": "Tracked mobile robot"},
            "Autonomous_Car": {"type": "Platform", "desc": "Autonomous vehicle"}
        }

        # 合并所有实体
        all_entities = {
            **algorithm_entities, **sensor_entities, **environment_entities,
            **task_entities, **evaluation_entities, **platform_entities
        }

        # 验证并添加实体
        for entity_name, entity_info in all_entities.items():
            if self.enable_schema_validation:
                valid, message = self.schema.validate_entity(
                    entity_name, entity_info["type"],
                    {"name": entity_name, "description": entity_info["desc"]}
                )
                if not valid:
                    self.stats["validation_errors"].append(message)
                    continue

            self.entities[entity_name] = entity_info
            self.stats["entities_by_type"][entity_info["type"]] += 1

        print(f"Defined {len(self.entities)} entities")
        return self.entities

    def define_triples(self):
        """定义关系三元组（200-300个）"""
        print("Defining triples...")

        # 生成三元组 - 确保有200-300个
        triples = []

        # used_in 关系: 算法应用于环境 (40个)
        used_in_relations = [
            ("A_star", "used_in", "Indoor"), ("Dijkstra", "used_in", "Indoor"),
            ("RRT", "used_in", "Outdoor"), ("RRT_star", "used_in", "Outdoor"),
            ("DWA", "used_in", "Dynamic"), ("Potential_Field", "used_in", "Dynamic"),
            ("SLAM", "used_in", "Unknown"), ("AMCL", "used_in", "Indoor"),
            ("EKF_SLAM", "used_in", "Structured"), ("FastSLAM", "used_in", "Unstructured"),
            ("PRM", "used_in", "Cluttered"), ("PRM_star", "used_in", "Cluttered"),
            ("FMT", "used_in", "Urban"), ("BIT_star", "used_in", "Urban"),
            ("Theta_star", "used_in", "Indoor"), ("Lazy_PRM", "used_in", "Industrial"),
            ("SBP", "used_in", "Structured"), ("D_star", "used_in", "Dynamic"),
            ("LPA_star", "used_in", "Dynamic"), ("AD_star", "used_in", "Dynamic"),
            ("A_star", "used_in", "Urban"), ("Dijkstra", "used_in", "Structured"),
            ("RRT", "used_in", "Aerial"), ("RRT_star", "used_in", "Aerial"),
            ("DWA", "used_in", "Industrial"), ("Potential_Field", "used_in", "Industrial"),
            ("SLAM", "used_in", "Outdoor"), ("AMCL", "used_in", "Urban"),
            ("EKF_SLAM", "used_in", "Industrial"), ("FastSLAM", "used_in", "Planetary"),
            ("PRM", "used_in", "Underwater"), ("PRM_star", "used_in", "Underwater"),
            ("FMT", "used_in", "Cluttered"), ("BIT_star", "used_in", "Cluttered"),
            ("Theta_star", "used_in", "Urban"), ("Lazy_PRM", "used_in", "Structured"),
            ("SBP", "used_in", "Indoor"), ("D_star", "used_in", "Urban"),
            ("LPA_star", "used_in", "Outdoor"), ("AD_star", "used_in", "Outdoor")
        ]
        triples.extend(used_in_relations)

        # requires 关系: 算法需要传感器 (40个)
        requires_relations = [
            ("A_star", "requires", "Lidar"), ("Dijkstra", "requires", "Lidar"),
            ("RRT", "requires", "Camera"), ("RRT_star", "requires", "Camera"),
            ("DWA", "requires", "Lidar"), ("Potential_Field", "requires", "Ultrasonic"),
            ("SLAM", "requires", "Lidar"), ("SLAM", "requires", "IMU"),
            ("AMCL", "requires", "Lidar"), ("EKF_SLAM", "requires", "Odometry"),
            ("FastSLAM", "requires", "Camera"), ("PRM", "requires", "Lidar"),
            ("PRM_star", "requires", "Lidar"), ("FMT", "requires", "RGB_D_Camera"),
            ("BIT_star", "requires", "Stereo_Camera"), ("Theta_star", "requires", "Lidar"),
            ("Lazy_PRM", "requires", "ToF_Camera"), ("SBP", "requires", "Radar"),
            ("D_star", "requires", "Sonar"), ("LPA_star", "requires", "Compass"),
            ("AD_star", "requires", "Encoder"), ("Path_Planning", "requires", "Lidar"),
            ("Obstacle_Avoidance", "requires", "Ultrasonic"), ("Localization", "requires", "GPS"),
            ("Mapping", "requires", "Lidar"), ("Navigation", "requires", "IMU"),
            ("Exploration", "requires", "Camera"), ("Tracking", "requires", "RGB_D_Camera"),
            ("Formation", "requires", "GPS"), ("Manipulation", "requires", "Force_Torque"),
            ("Inspection", "requires", "Camera"), ("A_star", "requires", "Odometry"),
            ("Dijkstra", "requires", "Odometry"), ("RRT", "requires", "IMU"),
            ("RRT_star", "requires", "IMU"), ("DWA", "requires", "Infrared"),
            ("Potential_Field", "requires", "Infrared"), ("SLAM", "requires", "GPS"),
            ("AMCL", "requires", "Odometry"), ("EKF_SLAM", "requires", "IMU")
        ]
        triples.extend(requires_relations)

        # evaluated_by 关系: 算法用指标评价 (40个)
        evaluated_by_relations = [
            ("A_star", "evaluated_by", "Path_Length"), ("Dijkstra", "evaluated_by", "Path_Length"),
            ("RRT", "evaluated_by", "Computation_Time"), ("RRT_star", "evaluated_by", "Path_Length"),
            ("DWA", "evaluated_by", "Safety_Margin"), ("Potential_Field", "evaluated_by", "Smoothness"),
            ("SLAM", "evaluated_by", "Accuracy"), ("AMCL", "evaluated_by", "Accuracy"),
            ("EKF_SLAM", "evaluated_by", "Robustness"), ("FastSLAM", "evaluated_by", "Success_Rate"),
            ("PRM", "evaluated_by", "Path_Length"), ("PRM_star", "evaluated_by", "Path_Length"),
            ("FMT", "evaluated_by", "Computation_Time"), ("BIT_star", "evaluated_by", "Path_Length"),
            ("Theta_star", "evaluated_by", "Path_Length"), ("Lazy_PRM", "evaluated_by", "Computation_Time"),
            ("SBP", "evaluated_by", "Success_Rate"), ("D_star", "evaluated_by", "Robustness"),
            ("LPA_star", "evaluated_by", "Computation_Time"), ("AD_star", "evaluated_by", "Success_Rate"),
            ("A_star", "evaluated_by", "Success_Rate"), ("Dijkstra", "evaluated_by", "Success_Rate"),
            ("RRT", "evaluated_by", "Success_Rate"), ("RRT_star", "evaluated_by", "Success_Rate"),
            ("DWA", "evaluated_by", "Success_Rate"), ("Potential_Field", "evaluated_by", "Success_Rate"),
            ("SLAM", "evaluated_by", "Computation_Time"), ("AMCL", "evaluated_by", "Computation_Time"),
            ("EKF_SLAM", "evaluated_by", "Accuracy"), ("FastSLAM", "evaluated_by", "Accuracy"),
            ("PRM", "evaluated_by", "Success_Rate"), ("PRM_star", "evaluated_by", "Success_Rate"),
            ("FMT", "evaluated_by", "Path_Length"), ("BIT_star", "evaluated_by", "Success_Rate"),
            ("Theta_star", "evaluated_by", "Smoothness"), ("Lazy_PRM", "evaluated_by", "Path_Length"),
            ("SBP", "evaluated_by", "Path_Length"), ("D_star", "evaluated_by", "Success_Rate"),
            ("LPA_star", "evaluated_by", "Success_Rate"), ("AD_star", "evaluated_by", "Path_Length")
        ]
        triples.extend(evaluated_by_relations)

        # performs 关系: 算法执行任务 (40个)
        performs_relations = [
            ("A_star", "performs", "Path_Planning"), ("Dijkstra", "performs", "Path_Planning"),
            ("RRT", "performs", "Path_Planning"), ("RRT_star", "performs", "Path_Planning"),
            ("DWA", "performs", "Obstacle_Avoidance"), ("Potential_Field", "performs", "Obstacle_Avoidance"),
            ("SLAM", "performs", "Mapping"), ("SLAM", "performs", "Localization"),
            ("AMCL", "performs", "Localization"), ("EKF_SLAM", "performs", "Mapping"),
            ("FastSLAM", "performs", "Mapping"), ("PRM", "performs", "Path_Planning"),
            ("PRM_star", "performs", "Path_Planning"), ("FMT", "performs", "Path_Planning"),
            ("BIT_star", "performs", "Path_Planning"), ("Theta_star", "performs", "Path_Planning"),
            ("Lazy_PRM", "performs", "Path_Planning"), ("SBP", "performs", "Path_Planning"),
            ("D_star", "performs", "Path_Planning"), ("LPA_star", "performs", "Path_Planning"),
            ("AD_star", "performs", "Path_Planning"), ("A_star", "performs", "Navigation"),
            ("Dijkstra", "performs", "Navigation"), ("RRT", "performs", "Exploration"),
            ("RRT_star", "performs", "Exploration"), ("DWA", "performs", "Navigation"),
            ("Potential_Field", "performs", "Navigation"), ("SLAM", "performs", "Exploration"),
            ("AMCL", "performs", "Navigation"), ("EKF_SLAM", "performs", "Navigation"),
            ("FastSLAM", "performs", "Exploration"), ("PRM", "performs", "Exploration"),
            ("PRM_star", "performs", "Exploration"), ("FMT", "performs", "Exploration"),
            ("BIT_star", "performs", "Exploration"), ("Theta_star", "performs", "Navigation"),
            ("Lazy_PRM", "performs", "Exploration"), ("SBP", "performs", "Navigation"),
            ("D_star", "performs", "Tracking"), ("LPA_star", "performs", "Tracking")
        ]
        triples.extend(performs_relations)

        # compared_with 关系: 算法对比 (40个)
        compared_with_relations = [
            ("A_star", "compared_with", "Dijkstra"), ("RRT", "compared_with", "RRT_star"),
            ("DWA", "compared_with", "Potential_Field"), ("SLAM", "compared_with", "AMCL"),
            ("EKF_SLAM", "compared_with", "FastSLAM"), ("PRM", "compared_with", "PRM_star"),
            ("FMT", "compared_with", "BIT_star"), ("Theta_star", "compared_with", "A_star"),
            ("Lazy_PRM", "compared_with", "PRM"), ("SBP", "compared_with", "A_star"),
            ("D_star", "compared_with", "A_star"), ("LPA_star", "compared_with", "A_star"),
            ("AD_star", "compared_with", "D_star"), ("A_star", "compared_with", "Theta_star"),
            ("Dijkstra", "compared_with", "Theta_star"), ("RRT", "compared_with", "PRM"),
            ("RRT_star", "compared_with", "PRM_star"), ("DWA", "compared_with", "SBP"),
            ("Potential_Field", "compared_with", "DWA"), ("SLAM", "compared_with", "EKF_SLAM"),
            ("AMCL", "compared_with", "FastSLAM"), ("EKF_SLAM", "compared_with", "SLAM"),
            ("FastSLAM", "compared_with", "AMCL"), ("PRM", "compared_with", "RRT"),
            ("PRM_star", "compared_with", "RRT_star"), ("FMT", "compared_with", "RRT"),
            ("BIT_star", "compared_with", "RRT_star"), ("Theta_star", "compared_with", "Dijkstra"),
            ("Lazy_PRM", "compared_with", "PRM_star"), ("SBP", "compared_with", "Dijkstra"),
            ("D_star", "compared_with", "LPA_star"), ("LPA_star", "compared_with", "AD_star"),
            ("AD_star", "compared_with", "D_star"), ("A_star", "compared_with", "PRM"),
            ("Dijkstra", "compared_with", "PRM"), ("RRT", "compared_with", "FMT"),
            ("RRT_star", "compared_with", "BIT_star"), ("DWA", "compared_with", "Theta_star"),
            ("Potential_Field", "compared_with", "Lazy_PRM"), ("SLAM", "compared_with", "FastSLAM")
        ]
        triples.extend(compared_with_relations)

        # 平台相关关系 (40个)
        platform_relations = [
            ("Wheeled_Robot", "used_in", "Indoor"), ("Wheeled_Robot", "used_in", "Urban"),
            ("Aerial_Drone", "used_in", "Outdoor"), ("Aerial_Drone", "used_in", "Aerial"),
            ("Humanoid", "used_in", "Indoor"), ("Humanoid", "used_in", "Industrial"),
            ("Tracked_Robot", "used_in", "Unstructured"), ("Tracked_Robot", "used_in", "Planetary"),
            ("Autonomous_Car", "used_in", "Urban"), ("Autonomous_Car", "used_in", "Structured"),
            ("Wheeled_Robot", "requires", "Lidar"), ("Wheeled_Robot", "requires", "Odometry"),
            ("Aerial_Drone", "requires", "GPS"), ("Aerial_Drone", "requires", "IMU"),
            ("Humanoid", "requires", "Camera"), ("Humanoid", "requires", "Force_Torque"),
            ("Tracked_Robot", "requires", "Lidar"), ("Tracked_Robot", "requires", "IMU"),
            ("Autonomous_Car", "requires", "Lidar"), ("Autonomous_Car", "requires", "Radar"),
            ("A_star", "performs", "Navigation"), ("Dijkstra", "performs", "Navigation"),
            ("RRT", "performs", "Exploration"), ("RRT_star", "performs", "Exploration"),
            ("DWA", "performs", "Obstacle_Avoidance"), ("Potential_Field", "performs", "Obstacle_Avoidance"),
            ("SLAM", "performs", "Mapping"), ("AMCL", "performs", "Localization"),
            ("EKF_SLAM", "performs", "Mapping"), ("FastSLAM", "performs", "Mapping"),
            ("Wheeled_Robot", "evaluated_by", "Path_Length"), ("Aerial_Drone", "evaluated_by", "Energy_Consumption"),
            ("Humanoid", "evaluated_by", "Accuracy"), ("Tracked_Robot", "evaluated_by", "Robustness"),
            ("Autonomous_Car", "evaluated_by", "Safety_Margin"), ("Wheeled_Robot", "compared_with", "Tracked_Robot"),
            ("Aerial_Drone", "compared_with", "Autonomous_Car"), ("Humanoid", "compared_with", "Wheeled_Robot"),
            ("Tracked_Robot", "compared_with", "Wheeled_Robot"), ("Autonomous_Car", "compared_with", "Aerial_Drone")
        ]
        triples.extend(platform_relations)

        # 验证并添加三元组
        for triple in triples:
            if self.enable_schema_validation:
                subject_type = self.entities[triple[0]]["type"]
                object_type = self.entities[triple[2]]["type"]

                valid, message = self.schema.validate_relation(subject_type, triple[1], object_type)
                if not valid:
                    self.stats["validation_errors"].append(message)
                    continue

            self.triples.append(triple)
            self.stats["relations_by_type"][triple[1]] += 1

        print(f"Defined {len(self.triples)} triples")
        return self.triples

    def build_knowledge_graph(self):
        """构建知识图谱网络"""
        print("Building knowledge graph...")
        self.graph = nx.Graph()

        # 添加节点
        for entity, info in self.entities.items():
            self.graph.add_node(entity, **info)

        # 添加边
        for subj, relation, obj in self.triples:
            self.graph.add_edge(subj, obj, relation=relation, weight=1.0)

        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph

    def generate_node2vec_embeddings(self, dimensions=128, walk_length=30, num_walks=200, workers=4):
        """使用Node2Vec生成实体嵌入"""
        print("Generating Node2Vec embeddings...")

        if self.graph is None:
            self.build_knowledge_graph()

        # 创建Node2Vec对象
        node2vec = Node2Vec(
            self.graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers
        )

        # 训练模型
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # 获取嵌入向量
        for entity in self.entities:
            if entity in model.wv:
                self.entity_embeddings[entity] = model.wv[entity]

        print(f"Generated embeddings for {len(self.entity_embeddings)} entities")
        return self.entity_embeddings

    def query_similar_entities(self, query_entity, top_k=5):
        """查询相似实体"""
        if query_entity not in self.entity_embeddings:
            return []

        query_embedding = self.entity_embeddings[query_entity].reshape(1, -1)
        similarities = {}

        for entity, embedding in self.entity_embeddings.items():
            if entity != query_entity:
                sim = cosine_similarity(query_embedding, embedding.reshape(1, -1))[0][0]
                similarities[entity] = {
                    'similarity': sim,
                    'type': self.entities[entity]['type']
                }

        # 按相似度排序并返回前k个
        sorted_similar = sorted(similarities.items(), key=lambda x: x[1]['similarity'], reverse=True)
        return sorted_similar[:top_k]

    def recommend_algorithms(self, environment, sensor_type, top_k=5):
        """基于环境和传感器推荐算法"""
        recommendations = []

        for algo, info in self.entities.items():
            if info["type"] == "Algorithm":
                score = 0

                # 检查算法是否适合该环境
                if (algo, "used_in", environment) in self.triples:
                    score += 1

                # 检查算法是否需要该传感器
                if (algo, "requires", sensor_type) in self.triples:
                    score += 1

                # 基于嵌入相似度的额外评分
                if algo in self.entity_embeddings:
                    # 查找在该环境中表现良好的算法
                    env_algorithms = [triple[0] for triple in self.triples
                                      if triple[1] == "used_in" and triple[2] == environment]

                    if env_algorithms:
                        # 计算与环境中其他算法的平均相似度
                        total_sim = 0
                        count = 0
                        for env_algo in env_algorithms:
                            if env_algo != algo and env_algo in self.entity_embeddings:
                                sim = cosine_similarity(
                                    self.entity_embeddings[algo].reshape(1, -1),
                                    self.entity_embeddings[env_algo].reshape(1, -1)
                                )[0][0]
                                total_sim += sim
                                count += 1

                        if count > 0:
                            score += total_sim / count

                if score > 0:
                    recommendations.append((algo, score))

        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_k]

    def visualize_knowledge_graph(self, figsize=(16, 12)):
        """可视化知识图谱"""
        if self.graph is None:
            self.build_knowledge_graph()

        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.graph, k=2, iterations=50)

        # 按实体类型着色
        node_colors = []
        color_map = {
            "Algorithm": "red",
            "Sensor": "blue",
            "Environment": "green",
            "Task": "orange",
            "Evaluation": "purple",
            "Platform": "brown"
        }

        for node in self.graph.nodes():
            node_type = self.entities[node]["type"]
            node_colors.append(color_map.get(node_type, "gray"))

        # 绘制节点和边
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors,
                               node_size=500, alpha=0.9)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, edge_color='gray', width=0.5)

        # 添加标签
        labels = {node: node.replace('_', '\n') for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=6)

        # 添加图例
        for entity_type, color in color_map.items():
            plt.scatter([], [], c=color, label=entity_type, alpha=0.8)
        plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc='best')

        plt.title("Robot Navigation Knowledge Graph", size=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def save_knowledge_graph(self, filepath="robot_navigation_kg"):
        """保存知识图谱数据"""
        os.makedirs(filepath, exist_ok=True)

        # 保存实体信息
        entity_data = []
        for entity, info in self.entities.items():
            entity_data.append({
                "entity": entity,
                "type": info["type"],
                "description": info["desc"],
                "embedding": self.entity_embeddings.get(entity, []).tolist() if entity in self.entity_embeddings else []
            })

        pd.DataFrame(entity_data).to_csv(f"{filepath}/entities.csv", index=False)

        # 保存三元组
        triple_data = [{"subject": s, "relation": r, "object": o} for s, r, o in self.triples]
        pd.DataFrame(triple_data).to_csv(f"{filepath}/triples.csv", index=False)

        # 保存Schema信息
        schema_stats = self.schema.get_schema_statistics()
        schema_df = pd.DataFrame([schema_stats])
        schema_df.to_csv(f"{filepath}/schema.csv", index=False)

        # 保存统计信息
        stats_df = pd.DataFrame([self.stats])
        stats_df.to_csv(f"{filepath}/statistics.csv", index=False)

        print(f"Knowledge graph data saved to {filepath}/")

    def print_statistics(self):
        """打印知识图谱统计信息"""
        print("\n=== Knowledge Graph Statistics ===")
        print(f"Total entities: {len(self.entities)}")
        print(f"Total triples: {len(self.triples)}")

        print("\nEntities by type:")
        for entity_type, count in self.stats["entities_by_type"].items():
            print(f"  {entity_type}: {count}")

        print("\nRelations by type:")
        for relation, count in self.stats["relations_by_type"].items():
            print(f"  {relation}: {count}")

        if self.graph:
            print(f"\nGraph density: {nx.density(self.graph):.4f}")
            print(f"Average degree: {sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes():.2f}")

        if self.entity_embeddings:
            print(f"Entities with embeddings: {len(self.entity_embeddings)}")

        if self.stats["validation_errors"]:
            print(f"\nValidation errors: {len(self.stats['validation_errors'])}")

    def build_complete_kg(self):
        """构建完整的知识图谱"""
        print("Building complete Robot Navigation Knowledge Graph...")
        self.define_entities()
        self.define_triples()
        self.build_knowledge_graph()
        self.generate_node2vec_embeddings()
        print("Knowledge Graph construction completed!")


# ==================== 使用示例 ====================

def main():
    """主函数示例"""
    print("=== 机器人导航知识图谱完整实现 ===")

    # 初始化知识图谱
    kg = RobotNavigationKnowledgeGraph(enable_schema_validation=True)

    # 构建完整图谱
    kg.build_complete_kg()

    # 打印统计信息
    kg.print_statistics()

    # 1. 相似实体查询
    print("\n=== 相似实体查询 ===")
    test_entities = ["A_star", "RRT", "Lidar", "Indoor"]

    for entity in test_entities:
        print(f"\n与 {entity} 相似的实体:")
        similar = kg.query_similar_entities(entity, top_k=5)
        for similar_entity, info in similar:
            print(f"  {similar_entity} ({info['type']}): {info['similarity']:.3f}")

    # 2. 算法推荐
    print("\n=== 算法推荐 ===")
    scenarios = [
        ("Indoor", "Lidar"),
        ("Outdoor", "GPS"),
        ("Dynamic", "Camera")
    ]

    for environment, sensor in scenarios:
        print(f"\n{environment} 环境使用 {sensor} 的推荐算法:")
        recommendations = kg.recommend_algorithms(environment, sensor)
        for algo, score in recommendations:
            print(f"  {algo}: 推荐分数 = {score:.3f}")

    # 3. 可视化
    print("\n=== 生成可视化 ===")
    kg.visualize_knowledge_graph()

    # 4. 保存数据
    print("\n=== 保存知识图谱数据 ===")
    kg.save_knowledge_graph()

    # 5. Schema信息
    print("\n=== Schema统计 ===")
    schema_stats = kg.schema.get_schema_statistics()
    print(f"实体类型: {schema_stats['entity_types']}")
    print(f"关系类型: {schema_stats['relation_types']}")


if __name__ == "__main__":
    # 安装依赖提示
    print("请确保已安装以下依赖:")
    print("pip install networkx pandas numpy matplotlib scikit-learn node2vec")
    print()

    # 运行主函数
    main()