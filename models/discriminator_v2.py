import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
import numpy as np
import random
from collections import defaultdict

class ImprovedMultiHeadDiscriminator(nn.Module):
    """
    Improved discriminator with:
    1. Relation features in GNN
    2. Attention pooling for core path
    3. Better feature fusion
    """
    
    def __init__(self, embedding_dim, hidden_dim=128, num_heads=3):
        super(ImprovedMultiHeadDiscriminator, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Edge feature dimension (for relation embeddings)
        self.edge_dim = embedding_dim
        
        # Improved GNN with edge features
        # GATv2 supports edge features better than standard GAT
        self.gnn_layers = nn.ModuleList([
            GATConv(embedding_dim, hidden_dim, heads=4, concat=True, edge_dim=self.edge_dim),
            GATConv(hidden_dim * 4, hidden_dim, heads=2, concat=True, edge_dim=self.edge_dim),
            GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        ])
        
        # Attention mechanism for core path vs context nodes
        self.path_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Three critique heads with deeper networks
        self.structural_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.semantic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for question fusion
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.logic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Question encoder for semantic evaluation
        self.question_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, graphlet_data, question_emb=None):
        """
        Forward pass with improved features
        """
        x = graphlet_data['node_features']  # [num_nodes, embedding_dim]
        edge_index = graphlet_data['edge_index']  # [2, num_edges]
        edge_attr = graphlet_data.get('edge_features', None)  # [num_edges, edge_dim]
        batch = graphlet_data.get('batch', None)
        core_path_mask = graphlet_data.get('core_path_mask', None)  # Boolean mask for core nodes
        
        # GNN encoding with edge features
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        # Attention-based pooling with emphasis on core path
        if core_path_mask is not None:
            # Calculate attention weights
            attention_scores = self.path_attention(x)  # [num_nodes, 1]
            
            # Apply higher weight to core path nodes
            attention_scores = attention_scores.squeeze(-1)  # [num_nodes]
            attention_scores[core_path_mask] += 2.0  # Boost core path importance
            
            attention_weights = F.softmax(attention_scores, dim=0)  # [num_nodes]
            
            # Weighted pooling
            if batch is not None:
                # Batch-wise attention pooling
                graphlet_repr = torch.zeros(batch.max().item() + 1, self.hidden_dim).to(x.device)
                for b in range(batch.max().item() + 1):
                    mask = (batch == b)
                    batch_weights = attention_weights[mask]
                    batch_weights = batch_weights / batch_weights.sum()
                    graphlet_repr[b] = (x[mask] * batch_weights.unsqueeze(-1)).sum(dim=0)
            else:
                attention_weights = attention_weights.unsqueeze(-1)  # [num_nodes, 1]
                graphlet_repr = (x * attention_weights).sum(dim=0, keepdim=True)  # [1, hidden_dim]
        else:
            # Fallback to global mean pooling
            if batch is not None:
                graphlet_repr = global_mean_pool(x, batch)
            else:
                graphlet_repr = torch.mean(x, dim=0, keepdim=True)
        
        # Multi-head evaluation
        structural_score = self.structural_head(graphlet_repr)
        logic_score = self.logic_head(graphlet_repr)
        
        # Enhanced semantic score with question
        if question_emb is not None:
            q_encoded = self.question_encoder(question_emb)
            if q_encoded.dim() == 1:
                q_encoded = q_encoded.unsqueeze(0)
            
            # Concatenate graphlet and question representations
            combined_repr = torch.cat([graphlet_repr, q_encoded], dim=-1)
            semantic_score = self.semantic_head(combined_repr)
        else:
            # Pad with zeros if no question
            padding = torch.zeros_like(graphlet_repr).to(graphlet_repr.device)
            combined_repr = torch.cat([graphlet_repr, padding], dim=-1)
            semantic_score = self.semantic_head(combined_repr)
        
        return {
            'structural': structural_score,
            'semantic': semantic_score,
            'logic': logic_score,
            'total': structural_score + semantic_score + logic_score
        }


class HardNegativeSampler:
    """
    Generate hard negative samples based on graph structure
    """
    
    def __init__(self, graph, entity_dict, embedding_manager):
        self.graph = graph
        self.entity_dict = entity_dict
        self.embedding_manager = embedding_manager
        
        # Pre-compute entity neighborhoods for faster sampling
        self.entity_neighbors = defaultdict(set)
        self.relation_targets = defaultdict(lambda: defaultdict(set))
        
        for source, target, data in graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            self.entity_neighbors[source].add(target)
            self.entity_neighbors[target].add(source)
            self.relation_targets[source][relation].add(target)
    
    def find_contextually_similar_entities(self, entity_id, answer_id, k=5):
        """
        Find entities that share context with the head entity
        """
        similar_entities = set()
        
        # Find entities connected through same relations
        if entity_id in self.entity_neighbors:
            # Get all neighbors of head entity
            head_neighbors = self.entity_neighbors[entity_id]
            
            # Find entities that share neighbors with head
            for neighbor in head_neighbors:
                if neighbor != answer_id:
                    # Get entities connected to this shared neighbor
                    for similar in self.entity_neighbors[neighbor]:
                        if similar != entity_id and similar != answer_id:
                            similar_entities.add(similar)
        
        return list(similar_entities)[:k]
    
    def generate_hard_negatives(self, golden_path, num_negatives=3):
        """
        Generate hard negative samples
        """
        head_id = golden_path['head_id']
        correct_relation = golden_path['relations'][0]
        correct_answer_id = golden_path['answer_id']
        negatives = []
        
        # Strategy 1: Contextually similar wrong answers
        similar_entities = self.find_contextually_similar_entities(head_id, correct_answer_id)
        for similar_entity in similar_entities[:1]:
            negatives.append({
                'head_id': head_id,
                'entities': [head_id, similar_entity],
                'relations': [correct_relation],
                'answer_id': similar_entity,
                'answer_name': self.entity_dict.get(similar_entity, 'unknown'),
                'type': 'hard_contextual'
            })
        
        # Strategy 2: Plausible but wrong relation
        # Find relations that also connect from head but to different targets
        if head_id in self.relation_targets:
            plausible_relations = [rel for rel in self.relation_targets[head_id].keys() 
                                  if rel != correct_relation and len(self.relation_targets[head_id][rel]) > 0]
            
            if plausible_relations:
                wrong_rel = random.choice(plausible_relations)
                wrong_target = random.choice(list(self.relation_targets[head_id][wrong_rel]))
                negatives.append({
                    'head_id': head_id,
                    'entities': [head_id, wrong_target],
                    'relations': [wrong_rel],
                    'answer_id': wrong_target,
                    'answer_name': self.entity_dict.get(wrong_target, 'unknown'),
                    'type': 'hard_relation'
                })
        
        # Strategy 3: Semantically similar entities (based on embeddings)
        if len(negatives) < num_negatives:
            # Get embedding of correct answer
            correct_emb = self.embedding_manager.get_entity_embedding(correct_answer_id)
            
            # Find semantically similar entities
            all_entities = list(self.entity_dict.keys())
            random_entities = random.sample(all_entities, min(100, len(all_entities)))
            
            similarities = []
            for entity_id in random_entities:
                if entity_id != head_id and entity_id != correct_answer_id:
                    entity_emb = self.embedding_manager.get_entity_embedding(entity_id)
                    similarity = F.cosine_similarity(correct_emb.unsqueeze(0), entity_emb.unsqueeze(0))
                    similarities.append((entity_id, similarity.item()))
            
            # Sort by similarity and take the most similar (but wrong) ones
            similarities.sort(key=lambda x: x[1], reverse=True)
            for similar_id, _ in similarities[:1]:
                negatives.append({
                    'head_id': head_id,
                    'entities': [head_id, similar_id],
                    'relations': [correct_relation],
                    'answer_id': similar_id,
                    'answer_name': self.entity_dict.get(similar_id, 'unknown'),
                    'type': 'hard_semantic'
                })
                if len(negatives) >= num_negatives:
                    break
        
        # Fallback to random if needed
        while len(negatives) < num_negatives:
            all_entities = list(self.entity_dict.keys())
            random_entity = random.choice([e for e in all_entities 
                                          if e != head_id and e != correct_answer_id])
            negatives.append({
                'head_id': head_id,
                'entities': [head_id, random_entity],
                'relations': [correct_relation],
                'answer_id': random_entity,
                'answer_name': self.entity_dict.get(random_entity, 'unknown'),
                'type': 'random'
            })
        
        return negatives[:num_negatives]


class ImprovedGraphletBuilder:
    """
    Build graphlets with relation features and core path marking
    """
    
    def __init__(self, dataset, embedding_manager=None):
        self.dataset = dataset
        self.entity_dict = dataset.entity_dict
        self.embedding_manager = embedding_manager
        self.entity_relations = dataset.entity_relations
    
    def build_evidence_graphlet(self, path):
        """
        Build evidence graphlet with improved features
        """
        entities = path['entities']
        relations = path['relations']
        
        # Collect all nodes and their relations
        all_nodes = set(entities)
        node_relations = []
        
        # Add first-hop neighbors for context
        for entity_id in entities:
            for relation, neighbor_id in self.dataset.entity_relations[entity_id]:
                all_nodes.add(neighbor_id)
                node_relations.append((entity_id, neighbor_id, relation))
        
        all_nodes = list(all_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Build node features
        node_features = []
        for node_id in all_nodes:
            if self.embedding_manager:
                embedding = self.embedding_manager.get_entity_embedding(node_id).clone()
            else:
                embedding = self.dataset.get_entity_embedding(node_id)
            node_features.append(embedding)
        
        node_features = torch.stack(node_features)
        
        # Build edge index and edge features
        edge_index = []
        edge_features = []
        
        for src_id, tgt_id, relation in node_relations:
            if src_id in node_to_idx and tgt_id in node_to_idx:
                src_idx = node_to_idx[src_id]
                tgt_idx = node_to_idx[tgt_id]
                edge_index.append([src_idx, tgt_idx])
                
                # Add relation embedding as edge feature
                if self.embedding_manager:
                    rel_emb = self.embedding_manager.get_relation_embedding(relation).clone()
                else:
                    # Fallback to random embedding
                    rel_emb = torch.randn(self.embedding_manager.embedding_dim)
                edge_features.append(rel_emb)
        
        if edge_index:
            edge_index = torch.LongTensor(edge_index).transpose(0, 1)
            edge_features = torch.stack(edge_features)
        else:
            edge_index = torch.LongTensor([[0], [0]])
            edge_features = torch.zeros((1, self.embedding_manager.embedding_dim))
        
        # Mark core path nodes
        core_path_indices = [node_to_idx[eid] for eid in entities]
        core_path_mask = torch.zeros(len(all_nodes), dtype=torch.bool)
        core_path_mask[core_path_indices] = True
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'core_path': core_path_indices,
            'core_path_mask': core_path_mask,
            'all_nodes': all_nodes,
            'node_to_idx': node_to_idx
        }