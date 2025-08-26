import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import pickle
import json
import os
from tqdm import tqdm

class OptimizedEmbeddingManager:
    """优化的嵌入管理器：使用SBERT统一编码所有内容"""
    
    def __init__(self, data_path="data", device='cuda', cache_path="embeddings_cache"):
        self.data_path = data_path
        self.device = device
        self.cache_path = cache_path
        
        # Load SBERT model for text encoding
        print("Loading SBERT model...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        if device == 'cuda' and torch.cuda.is_available():
            self.sbert_model.to('cuda')
        
        # SBERT dimension (all-MiniLM-L6-v2 outputs 384-dim embeddings)
        self.embedding_dim = 384
        self.entity_embedding_dim = self.embedding_dim
        self.relation_embedding_dim = self.embedding_dim
        
        # Load components
        self.load_knowledge_graph()
        
        # Check for cached embeddings
        os.makedirs(cache_path, exist_ok=True)
        self.entity_cache_file = f"{cache_path}/entity_embeddings.pkl"
        self.relation_cache_file = f"{cache_path}/relation_embeddings.pkl"
        
        # Load or build embeddings
        if os.path.exists(self.entity_cache_file):
            print("Loading cached entity embeddings...")
            self.load_cached_entity_embeddings()
        else:
            self.build_entity_embeddings()
            self.cache_entity_embeddings()
            
        if os.path.exists(self.relation_cache_file):
            print("Loading cached relation embeddings...")
            self.load_cached_relation_embeddings()
        else:
            self.build_relation_embeddings()
            self.cache_relation_embeddings()
        
        # Question cache for runtime
        self.question_cache = {}
    
    def load_knowledge_graph(self):
        """加载知识图谱"""
        with open('graph/knowledge_graph.pkl', 'rb') as f:
            self.graph = pickle.load(f)
        
        # Extract all relations
        relations = set()
        for _, _, data in self.graph.edges(data=True):
            relations.add(data.get('relation', 'unknown'))
        
        self.relations = sorted(list(relations))
        print(f"Found {len(self.relations)} relation types: {self.relations}")
    
    def build_entity_embeddings(self):
        """使用SBERT编码实体名称"""
        print("Building entity embeddings with SBERT...")
        
        # Load entity dictionary
        with open('graph/entity_dict.json', 'r', encoding='utf-8') as f:
            entity_dict_str = json.load(f)
            self.entity_dict = {int(k): v for k, v in entity_dict_str.items()}
        
        # Prepare entity texts for encoding
        entity_texts = []
        entity_ids = []
        for entity_id, entity_name in self.entity_dict.items():
            entity_text = self.entity_to_text(entity_name)
            entity_texts.append(entity_text)
            entity_ids.append(entity_id)
        
        # Encode with SBERT in larger batches for efficiency
        print(f"Encoding {len(entity_texts)} entities with SBERT...")
        batch_size = 512 if self.device == 'cuda' else 256
        
        all_embeddings = []
        for i in tqdm(range(0, len(entity_texts), batch_size), desc="Encoding entities"):
            batch_texts = entity_texts[i:i+batch_size]
            with torch.no_grad():
                batch_embeddings = self.sbert_model.encode(
                    batch_texts, 
                    convert_to_tensor=True,
                    device='cuda' if self.device == 'cuda' and torch.cuda.is_available() else 'cpu',
                    batch_size=batch_size,
                    show_progress_bar=False
                )
                all_embeddings.append(batch_embeddings.cpu())
        
        # Concatenate all embeddings
        entity_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Store embeddings
        self.entity_embeddings = {}
        for i, entity_id in enumerate(entity_ids):
            self.entity_embeddings[entity_id] = entity_embeddings[i].numpy()
        
        print(f"Built {len(self.entity_embeddings)} entity embeddings, dim={self.embedding_dim}")
    
    def build_relation_embeddings(self):
        """使用SBERT编码关系名称"""
        print("Building relation embeddings with SBERT...")
        
        # Prepare relation texts for encoding
        relation_texts = []
        for relation in self.relations:
            relation_text = self.relation_to_text(relation)
            relation_texts.append(relation_text)
        
        # Encode with SBERT
        with torch.no_grad():
            relation_embeddings = self.sbert_model.encode(
                relation_texts, 
                convert_to_tensor=True,
                device='cuda' if self.device == 'cuda' and torch.cuda.is_available() else 'cpu'
            )
        
        # Store embeddings
        self.relation_embeddings = {}
        self.relation_to_id = {}
        self.id_to_relation = {}
        
        for i, relation in enumerate(self.relations):
            self.relation_embeddings[relation] = relation_embeddings[i].cpu().numpy()
            self.relation_to_id[relation] = i
            self.id_to_relation[i] = relation
        
        print(f"Built relation embeddings, dim={self.relation_embedding_dim}")
    
    def cache_entity_embeddings(self):
        """缓存实体嵌入"""
        cache_data = {
            'entity_embeddings': self.entity_embeddings,
            'entity_dict': self.entity_dict,
            'embedding_dim': self.embedding_dim
        }
        with open(self.entity_cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cached entity embeddings to {self.entity_cache_file}")
    
    def load_cached_entity_embeddings(self):
        """加载缓存的实体嵌入"""
        with open(self.entity_cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        self.entity_embeddings = cache_data['entity_embeddings']
        self.entity_dict = cache_data['entity_dict']
        print(f"Loaded {len(self.entity_embeddings)} cached entity embeddings")
    
    def cache_relation_embeddings(self):
        """缓存关系嵌入"""
        cache_data = {
            'relation_embeddings': self.relation_embeddings,
            'relation_to_id': self.relation_to_id,
            'id_to_relation': self.id_to_relation,
            'embedding_dim': self.relation_embedding_dim
        }
        with open(self.relation_cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cached relation embeddings to {self.relation_cache_file}")
    
    def load_cached_relation_embeddings(self):
        """加载缓存的关系嵌入"""
        with open(self.relation_cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        self.relation_embeddings = cache_data['relation_embeddings']
        self.relation_to_id = cache_data['relation_to_id']
        self.id_to_relation = cache_data['id_to_relation']
        print(f"Loaded {len(self.relation_embeddings)} cached relation embeddings")
    
    def entity_to_text(self, entity_name):
        """将实体名称转换为更自然的文本格式"""
        return entity_name.replace('_', ' ').strip()
    
    def relation_to_text(self, relation):
        """将关系名称转换为自然语言文本"""
        relation_map = {
            'directed_by': 'directed by',
            'starred_actors': 'starred actors',
            'written_by': 'written by',
            'has_genre': 'has genre',
            'has_tags': 'has tags',
            'release_year': 'released in year',
            'in_language': 'in language',
            'has_imdb_rating': 'has IMDB rating',
            'has_imdb_votes': 'has IMDB votes'
        }
        return relation_map.get(relation, relation.replace('_', ' '))
    
    def encode_question(self, question_text):
        """使用SBERT编码问题（带缓存）"""
        if question_text in self.question_cache:
            return self.question_cache[question_text]
        
        with torch.no_grad():
            question_emb = self.sbert_model.encode(
                question_text,
                convert_to_tensor=True,
                device='cuda' if self.device == 'cuda' and torch.cuda.is_available() else 'cpu'
            )
        
        self.question_cache[question_text] = question_emb.cpu()
        return question_emb.cpu()
    
    def get_entity_embedding(self, entity_id):
        """获取实体嵌入"""
        if entity_id in self.entity_embeddings:
            return torch.FloatTensor(self.entity_embeddings[entity_id])
        else:
            # For unknown entities, encode on the fly
            if entity_id in self.entity_dict:
                entity_text = self.entity_to_text(self.entity_dict[entity_id])
                with torch.no_grad():
                    entity_emb = self.sbert_model.encode(
                        entity_text,
                        convert_to_tensor=True,
                        device='cuda' if self.device == 'cuda' and torch.cuda.is_available() else 'cpu'
                    )
                    self.entity_embeddings[entity_id] = entity_emb.cpu().numpy()
                    return entity_emb.cpu()
            return torch.zeros(self.embedding_dim)
    
    def get_relation_embedding(self, relation_name):
        """获取关系嵌入"""
        if relation_name in self.relation_embeddings:
            return torch.FloatTensor(self.relation_embeddings[relation_name])
        else:
            return torch.zeros(self.relation_embedding_dim)
    
    def get_relation_id(self, relation_name):
        """获取关系ID"""
        return self.relation_to_id.get(relation_name, 0)
    
    def save_embeddings(self, save_path="embeddings"):
        """保存所有嵌入"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save relation embeddings
        relation_data = {
            'embeddings': self.relation_embeddings,
            'relation_to_id': self.relation_to_id,
            'id_to_relation': self.id_to_relation,
            'dimension': self.relation_embedding_dim
        }
        
        with open(f"{save_path}/relation_embeddings.pkl", 'wb') as f:
            pickle.dump(relation_data, f)
        
        print(f"Saved embeddings to {save_path}")

def test_embeddings():
    """测试嵌入管理器"""
    print("Testing Optimized Embedding Manager...")
    
    # Create embedding manager
    embedding_manager = OptimizedEmbeddingManager(device='cpu')
    
    # Test question encoding
    sample_questions = [
        "what movies are about [Top Hat]",
        "which films are directed by [Clint Eastwood]",
        "what is the genre of [Drama]"
    ]
    
    for question in sample_questions:
        question_emb = embedding_manager.encode_question(question)
        print(f"Question: '{question}' -> embedding shape: {question_emb.shape}")
    
    # Test relation embeddings
    print(f"\nRelation embeddings:")
    for relation in embedding_manager.relations[:5]:
        rel_emb = embedding_manager.get_relation_embedding(relation)
        print(f"Relation: '{relation}' -> embedding shape: {rel_emb.shape}")
    
    # Test entity embeddings
    print(f"\nEntity embeddings:")
    for entity_id in list(embedding_manager.entity_dict.keys())[:5]:
        entity_emb = embedding_manager.get_entity_embedding(entity_id)
        entity_name = embedding_manager.entity_dict[entity_id]
        print(f"Entity: '{entity_name}' (ID: {entity_id}) -> embedding shape: {entity_emb.shape}")

if __name__ == "__main__":
    test_embeddings()