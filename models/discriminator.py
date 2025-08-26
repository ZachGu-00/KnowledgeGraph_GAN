import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import numpy as np
import random
from collections import defaultdict

class MultiHeadDiscriminator(nn.Module):
    """
    Multi-head discriminator that evaluates evidence graphlets from three perspectives:
    1. Structural coherence
    2. Semantic relevance  
    3. Logical flow
    """
    
    def __init__(self, embedding_dim, hidden_dim=128, num_heads=3):
        super(MultiHeadDiscriminator, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Shared GNN encoder for graphlet representation
        self.gnn_layers = nn.ModuleList([
            GATConv(embedding_dim, hidden_dim, heads=4, concat=True),
            GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        ])
        
        # Three critique heads
        self.structural_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.semantic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.logic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Question encoding for semantic evaluation
        self.question_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, graphlet_data, question_emb=None):
        """
        Forward pass through discriminator
        
        Args:
            graphlet_data: Dict containing graphlet information
                - node_features: [num_nodes, embedding_dim]
                - edge_index: [2, num_edges] 
                - batch: batch assignment for nodes (optional)
            question_emb: Question embedding [embedding_dim]
            
        Returns:
            scores: Dict with scores from each head
        """
        x = graphlet_data['node_features']  # [num_nodes, embedding_dim]
        edge_index = graphlet_data['edge_index']  # [2, num_edges]
        batch = graphlet_data.get('batch', None)
        
        # GNN encoding
        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
        
        # Global pooling to get graphlet representation
        if batch is not None:
            graphlet_repr = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        else:
            graphlet_repr = torch.mean(x, dim=0, keepdim=True)  # [1, hidden_dim]
        
        # Multi-head evaluation
        structural_score = self.structural_head(graphlet_repr)
        semantic_score = self.semantic_head(graphlet_repr)
        logic_score = self.logic_head(graphlet_repr)
        
        # Incorporate question context for semantic score
        if question_emb is not None:
            q_encoded = self.question_encoder(question_emb)  # [hidden_dim]
            if q_encoded.dim() == 1:
                q_encoded = q_encoded.unsqueeze(0)  # [1, hidden_dim]
            
            # Attention between graphlet and question
            attention = torch.matmul(graphlet_repr, q_encoded.transpose(-2, -1))  # [batch, 1]
            attention = F.softmax(attention, dim=-1)
            semantic_score = semantic_score * attention
        
        return {
            'structural': structural_score,
            'semantic': semantic_score, 
            'logic': logic_score,
            'total': structural_score + semantic_score + logic_score
        }

class GraphletBuilder:
    """Helper class to build evidence graphlets from paths"""
    
    def __init__(self, dataset, embedding_manager=None):
        self.dataset = dataset
        self.entity_dict = dataset.entity_dict
        self.embedding_manager = embedding_manager  # Use SBERT embeddings
        self.entity_relations = dataset.entity_relations
        
    def build_evidence_graphlet(self, path):
        """
        Build evidence graphlet from a reasoning path
        
        Args:
            path: Dict with 'entities', 'relations' keys
            
        Returns:
            graphlet_data: Dict for PyTorch Geometric
        """
        entities = path['entities']  # [head_id, ..., answer_id]
        relations = path['relations']  # [rel1, rel2, ...]
        
        # Collect all nodes (entities + first-hop neighbors)
        all_nodes = set(entities)
        node_relations = []  # For building edges
        
        # Add first-hop neighbors for context
        for entity_id in entities:
            for relation, neighbor_id in self.dataset.entity_relations[entity_id]:
                all_nodes.add(neighbor_id)
                node_relations.append((entity_id, neighbor_id, relation))
        
        # Convert to lists
        all_nodes = list(all_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Build node features (entity embeddings) with dimension checking
        node_features = []
        for node_id in all_nodes:
            if self.embedding_manager:
                # Use SBERT embeddings (384 dim)
                embedding = self.embedding_manager.get_entity_embedding(node_id).clone()
            else:
                # Fallback to old embeddings
                embedding = self.dataset.get_entity_embedding(node_id)
                if embedding.dim() == 0:
                    embedding = embedding.unsqueeze(0)
                elif embedding.dim() > 1:
                    embedding = embedding.flatten()
            node_features.append(embedding)
        
        # Ensure all embeddings have the same dimension before stacking
        if node_features:
            # Get all dimensions
            dims = [emb.shape[0] for emb in node_features]
            if len(set(dims)) > 1:
                # Pad to maximum dimension
                max_dim = max(dims)
                padded_features = []
                for emb in node_features:
                    if emb.shape[0] < max_dim:
                        padding = torch.zeros(max_dim - emb.shape[0])
                        emb = torch.cat([emb, padding])
                    padded_features.append(emb)
                node_features = torch.stack(padded_features)
            else:
                node_features = torch.stack(node_features)
        else:
            # Empty case
            node_features = torch.zeros((1, self.dataset.embedding_dim))
        
        # Build edge index
        edge_index = []
        for src_id, tgt_id, relation in node_relations:
            if src_id in node_to_idx and tgt_id in node_to_idx:
                src_idx = node_to_idx[src_id]
                tgt_idx = node_to_idx[tgt_id]
                edge_index.append([src_idx, tgt_idx])
        
        if edge_index:
            edge_index = torch.LongTensor(edge_index).transpose(0, 1)  # [2, num_edges]
        else:
            edge_index = torch.LongTensor([[0], [0]])  # Dummy edge
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'core_path': [node_to_idx[eid] for eid in entities],
            'all_nodes': all_nodes,
            'node_to_idx': node_to_idx
        }
    
    def corrupt_graphlet(self, golden_path, corruption_type='relation'):
        """
        Create negative samples by corrupting golden paths
        
        Args:
            golden_path: Golden path dict
            corruption_type: 'relation', 'answer', or 'incomplete'
            
        Returns:
            corrupted_path: Corrupted path dict
        """
        head_id = golden_path['head_id']
        relations = golden_path['relations'].copy()
        entities = golden_path['entities'].copy()
        
        if corruption_type == 'relation' and len(relations) > 0:
            # Replace a random relation with a wrong one
            valid_relations = self.dataset.get_valid_relations(head_id)
            if valid_relations:
                wrong_relations = [r for r in valid_relations if r != relations[0]]
                if wrong_relations:
                    relations[0] = random.choice(wrong_relations)
                    
                    # Find where this wrong relation leads
                    for rel, target_id in self.dataset.entity_relations[head_id]:
                        if rel == relations[0]:
                            entities[1] = target_id
                            break
        
        elif corruption_type == 'answer':
            # Keep path structure but replace final answer
            if len(entities) > 1:
                all_entities = list(self.entity_dict.keys())
                wrong_answers = [e for e in all_entities if e != entities[-1]]
                if wrong_answers:
                    entities[-1] = random.choice(wrong_answers)
        
        elif corruption_type == 'incomplete':
            # Truncate the path (for multi-hop)
            if len(entities) > 2:
                entities = entities[:-1]
                relations = relations[:-1]
        
        return {
            'head_id': head_id,
            'relations': relations,
            'entities': entities,
            'answer_id': entities[-1] if entities else head_id,
            'answer_name': self.entity_dict.get(entities[-1], 'unknown') if entities else 'unknown'
        }

class DiscriminatorPretrainer:
    """Pretrainer for the discriminator using golden paths and corrupted samples"""
    
    def __init__(self, discriminator, dataset, device='cuda', embedding_manager=None):
        self.discriminator = discriminator
        self.dataset = dataset
        self.device = device
        self.embedding_manager = embedding_manager
        self.graphlet_builder = GraphletBuilder(dataset, embedding_manager)
        
        self.discriminator.to(device)
        
        # Loss function: margin-based ranking loss
        self.margin = 1.0
        self.criterion = nn.MarginRankingLoss(margin=self.margin)
        
        # Head weights
        self.head_weights = {
            'structural': 0.3,
            'semantic': 0.4, 
            'logic': 0.3
        }
    
    def create_training_samples(self, batch, embedding_manager=None):
        """Create positive and negative samples from a batch"""
        positive_samples = []
        negative_samples = []
        questions = []
        
        # Use embedding_manager from instance if not provided
        if embedding_manager is None:
            embedding_manager = self.embedding_manager
        
        if embedding_manager is None:
            print("Warning: No embedding_manager available for discriminator")
            return [], [], []
        
        for i, golden_paths in enumerate(batch['golden_paths']):
            if not golden_paths:
                continue
                
            question = batch['questions'][i]
            # Use SBERT to encode question
            question_emb = embedding_manager.encode_question(question).clone()
            
            # Use first golden path as positive sample
            golden_path = golden_paths[0]
            positive_graphlet = self.graphlet_builder.build_evidence_graphlet(golden_path)
            positive_samples.append(positive_graphlet)
            questions.append(question_emb)
            
            # Generate negative samples
            corruption_types = ['relation', 'answer']
            for corruption_type in corruption_types:
                negative_path = self.graphlet_builder.corrupt_graphlet(golden_path, corruption_type)
                negative_graphlet = self.graphlet_builder.build_evidence_graphlet(negative_path)
                negative_samples.append(negative_graphlet)
        
        return positive_samples, negative_samples, questions
    
    def compute_loss(self, pos_scores, neg_scores):
        """Compute multi-head margin ranking loss"""
        total_loss = 0
        losses = {}
        
        for head in ['structural', 'semantic', 'logic']:
            pos_head_scores = pos_scores[head]  # [batch_size, 1]
            neg_head_scores = neg_scores[head]  # [batch_size * num_neg, 1]
            
            # Expand positive scores to match negative scores
            num_neg_per_pos = len(neg_head_scores) // len(pos_head_scores)
            expanded_pos = pos_head_scores.repeat(num_neg_per_pos, 1)  # [batch_size * num_neg, 1]
            
            # Target: positive should be ranked higher than negative
            target = torch.ones(len(expanded_pos)).to(self.device)
            
            loss = self.criterion(expanded_pos.squeeze(), neg_head_scores.squeeze(), target)
            losses[head] = loss
            total_loss += self.head_weights[head] * loss
        
        return total_loss, losses
    
    def train_epoch(self, dataloader, optimizer):
        """Train discriminator for one epoch"""
        self.discriminator.train()
        total_loss = 0
        num_batches = 0
        
        if self.embedding_manager is None:
            print("Warning: No embedding_manager available for discriminator training")
            return 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            
            optimizer.zero_grad()
            
            try:
                # Create training samples
                pos_samples, neg_samples, questions = self.create_training_samples(batch)
                
                if not pos_samples or not neg_samples:
                    if batch_idx < 10:  # Only show first few warnings
                        print(f"Batch {batch_idx}: No valid samples (pos={len(pos_samples)}, neg={len(neg_samples)})")
                    continue
                
                if batch_idx == 0:  # Debug first batch
                    print(f"Discriminator: Processing batch with {len(pos_samples)} pos, {len(neg_samples)} neg samples")
                
                # Forward pass on positive samples
                pos_scores_list = []
                for i, pos_sample in enumerate(pos_samples):
                    # Move to device
                    for key in pos_sample:
                        if isinstance(pos_sample[key], torch.Tensor):
                            pos_sample[key] = pos_sample[key].to(self.device)
                    
                    question_emb = questions[i].clone().to(self.device) if i < len(questions) else None
                    pos_scores = self.discriminator(pos_sample, question_emb)
                    pos_scores_list.append(pos_scores)
                
                # Forward pass on negative samples  
                neg_scores_list = []
                for neg_sample in neg_samples:
                    # Move to device
                    for key in neg_sample:
                        if isinstance(neg_sample[key], torch.Tensor):
                            neg_sample[key] = neg_sample[key].to(self.device)
                    
                    neg_scores = self.discriminator(neg_sample)
                    neg_scores_list.append(neg_scores)
                
                # Combine scores
                pos_scores_combined = {}
                neg_scores_combined = {}
                
                for head in ['structural', 'semantic', 'logic']:
                    pos_scores_combined[head] = torch.cat([s[head] for s in pos_scores_list], dim=0)
                    neg_scores_combined[head] = torch.cat([s[head] for s in neg_scores_list], dim=0)
                
                # Compute loss
                loss, head_losses = self.compute_loss(pos_scores_combined, neg_scores_combined)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 1000 == 0:  # Less frequent logging
                    print(f"Disc Batch {batch_idx}, Loss: {loss.item():.4f}")
                    for head, head_loss in head_losses.items():
                        print(f"  {head}: {head_loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        return total_loss / max(num_batches, 1)

def test_discriminator():
    """Test discriminator implementation"""
    from data_loader import create_data_loaders
    
    print("Testing Discriminator...")
    
    # Load data
    datasets, loaders = create_data_loaders(hop="1hop", batch_size=4)
    train_dataset = datasets['train']
    train_loader = loaders['train']
    
    # Create discriminator
    embedding_dim = train_dataset.embedding_dim
    discriminator = MultiHeadDiscriminator(embedding_dim)
    
    # Test with one batch
    for batch in train_loader:
        if batch is not None:
            # Create pretrainer
            pretrainer = DiscriminatorPretrainer(discriminator, train_dataset, device='cpu')
            
            # Create samples
            pos_samples, neg_samples, questions = pretrainer.create_training_samples(batch)
            
            print(f"Created {len(pos_samples)} positive and {len(neg_samples)} negative samples")
            
            if pos_samples:
                pos_sample = pos_samples[0]
                print(f"Sample graphlet: {pos_sample['node_features'].shape} nodes")
                
                # Test forward pass
                scores = discriminator(pos_sample)
                print(f"Scores: {scores}")
            
            break
    
    print("Discriminator test completed!")

if __name__ == "__main__":
    test_discriminator()