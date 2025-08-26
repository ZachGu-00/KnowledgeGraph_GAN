import numpy as np
import torch
import json
import pickle
import networkx as nx
import re
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class MetaQADataset(Dataset):
    def __init__(self, data_path="data", query_path="query", hop="1hop", split="train"):
        self.data_path = Path(data_path)
        self.query_path = Path(query_path) 
        self.hop = hop
        self.split = split
        
        # Load components
        self.load_knowledge_graph()
        self.load_entity_embeddings()
        self.load_qa_data()
        self.build_relation_mappings()
        
        print(f"Loaded {len(self.qa_pairs)} QA pairs for {hop}-{split}")
    
    def load_knowledge_graph(self):
        """Load knowledge graph from kb.txt and the built graph"""
        print("Loading knowledge graph...")
        
        # Load the built graph
        with open('graph/knowledge_graph.pkl', 'rb') as f:
            self.graph = pickle.load(f)
            
        # Load entity dictionary
        with open('graph/entity_dict.json', 'r', encoding='utf-8') as f:
            entity_dict_str = json.load(f)
            self.entity_dict = {int(k): v for k, v in entity_dict_str.items()}
            self.entity_name_to_id = {v: int(k) for k, v in entity_dict_str.items()}
        
        # Build adjacency info for faster lookup
        self.entity_relations = defaultdict(list)  # entity_id -> [(relation, target_id), ...]
        
        for source, target, data in self.graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            self.entity_relations[source].append((relation, target))
        
        print(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def load_entity_embeddings(self):
        """Load entity embeddings from kb_entity.npz"""
        print("Loading entity embeddings...")
        
        embedding_file = self.data_path / "kb_entity.npz"
        self.embeddings_raw = np.load(embedding_file)
        
        # Create embedding matrix
        self.entity_embeddings = {}
        embedding_dim = None
        
        for key in self.embeddings_raw.files:
            try:
                entity_id = int(key.split('-')[1])
                if entity_id in self.entity_dict:
                    embedding = self.embeddings_raw[key]
                    # Flatten the embedding if it's 2D
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()
                    
                    self.entity_embeddings[entity_id] = embedding
                    if embedding_dim is None:
                        embedding_dim = len(embedding)
            except:
                continue
        
        self.embedding_dim = embedding_dim
        # Note: These are old MetaQA embeddings, we use SBERT instead
        # print(f"Loaded embeddings for {len(self.entity_embeddings)} entities, dim={embedding_dim}")
    
    def load_qa_data(self):
        """Load QA pairs from query files"""
        print(f"Loading QA data for {self.hop}-{self.split}...")
        
        qa_file = self.query_path / self.hop / f"qa_{self.split}.txt"
        self.qa_pairs = []
        
        with open(qa_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if '\t' in line:
                    question, answer = line.split('\t', 1)
                    
                    # Extract head entity from question (in brackets)
                    head_entity_match = re.search(r'\[([^\]]+)\]', question)
                    if head_entity_match:
                        head_entity = head_entity_match.group(1)
                        
                        # Parse multiple answers
                        answers = [a.strip() for a in answer.split('|')]
                        
                        self.qa_pairs.append({
                            'id': line_idx,
                            'question': question,
                            'head_entity': head_entity,
                            'answers': answers,
                            'raw_line': line
                        })
        
        print(f"Loaded {len(self.qa_pairs)} QA pairs")
    
    def build_relation_mappings(self):
        """Build relation type mappings"""
        relations = set()
        for _, _, data in self.graph.edges(data=True):
            relations.add(data.get('relation', 'unknown'))
        
        self.relations = sorted(list(relations))
        self.relation_to_id = {rel: i for i, rel in enumerate(self.relations)}
        self.id_to_relation = {i: rel for i, rel in enumerate(self.relations)}
        
        print(f"Found {len(self.relations)} relation types: {self.relations}")
    
    def find_golden_path(self, head_entity, answer_entities):
        """Find golden path(s) from head entity to answer entities in the KG"""
        head_id = self.entity_name_to_id.get(head_entity)
        if head_id is None:
            return None
        
        paths = []
        
        for answer in answer_entities:
            answer_id = self.entity_name_to_id.get(answer)
            if answer_id is None:
                continue
            
            # For 1-hop: check direct connection
            if self.hop == "1hop":
                for relation, target_id in self.entity_relations[head_id]:
                    if target_id == answer_id:
                        paths.append({
                            'head_id': head_id,
                            'relations': [relation],
                            'entities': [head_id, answer_id],
                            'answer_id': answer_id,
                            'answer_name': answer
                        })
        
        return paths if paths else None
    
    def get_entity_embedding(self, entity_id):
        """Get embedding for entity"""
        if entity_id in self.entity_embeddings:
            return torch.FloatTensor(self.entity_embeddings[entity_id])
        else:
            # Return zero embedding if not found
            return torch.zeros(self.embedding_dim)
    
    def get_valid_relations(self, entity_id):
        """Get all valid relations from an entity"""
        return [rel for rel, _ in self.entity_relations[entity_id]]
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        qa_pair = self.qa_pairs[idx]
        
        # Find golden path
        golden_paths = self.find_golden_path(qa_pair['head_entity'], qa_pair['answers'])
        
        return {
            'id': qa_pair['id'],
            'question': qa_pair['question'],
            'head_entity': qa_pair['head_entity'],
            'answers': qa_pair['answers'],
            'golden_paths': golden_paths,
            'head_id': self.entity_name_to_id.get(qa_pair['head_entity']),
        }

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    # Filter out samples without golden paths
    valid_batch = [item for item in batch if item['golden_paths'] is not None]
    
    if not valid_batch:
        return None
    
    return {
        'ids': [item['id'] for item in valid_batch],
        'questions': [item['question'] for item in valid_batch],
        'head_entities': [item['head_entity'] for item in valid_batch],
        'answers': [item['answers'] for item in valid_batch],
        'golden_paths': [item['golden_paths'] for item in valid_batch],
        'head_ids': [item['head_id'] for item in valid_batch],
    }

def create_data_loaders(data_path="data", query_path="query", hop="1hop", batch_size=32):
    """Create train/dev/test data loaders"""
    
    datasets = {}
    loaders = {}
    
    for split in ['train', 'dev', 'test']:
        try:
            datasets[split] = MetaQADataset(data_path, query_path, hop, split)
            loaders[split] = DataLoader(
                datasets[split], 
                batch_size=batch_size, 
                shuffle=(split=='train'),
                collate_fn=collate_fn,
                drop_last=False
            )
        except FileNotFoundError:
            print(f"Warning: {split} file not found for {hop}")
            continue
    
    return datasets, loaders

if __name__ == "__main__":
    # Test the data loader
    datasets, loaders = create_data_loaders(hop="1hop", batch_size=4)
    
    print("Testing data loader...")
    train_dataset = datasets['train']
    train_loader = loaders['train']
    
    # Show some statistics
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Embedding dim: {train_dataset.embedding_dim}")
    print(f"Relations: {train_dataset.relations[:5]}...")
    
    # Test one batch
    for batch_idx, batch in enumerate(train_loader):
        if batch is not None:
            print(f"Batch {batch_idx}:")
            print(f"  Questions: {len(batch['questions'])}")
            print(f"  Sample question: {batch['questions'][0]}")
            if batch['golden_paths'][0]:
                print(f"  Sample golden path: {batch['golden_paths'][0][0]}")
            break
        else:
            print(f"Skipping empty batch {batch_idx}")
            
    print("Data loader test completed!")