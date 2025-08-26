import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
import numpy as np
import random
from collections import defaultdict

class PathGenerator(nn.Module):
    """
    Path Constructor (Generator G): RL agent that constructs reasoning paths
    """
    
    def __init__(self, embedding_dim, num_relations, hidden_dim=128, num_layers=2):
        super(PathGenerator, self).__init__()
        
        # All embeddings now have the same dimension (SBERT: 384)
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Question encoder (SBERT embeddings -> hidden)
        self.question_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.3)
        )
        
        # Path encoder (for current path state)
        self.path_encoder = nn.LSTM(
            input_size=embedding_dim * 2,  # entity + relation embeddings (both SBERT)
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # GAT layers for policy network
        self.gat_layers = nn.ModuleList([
            GATConv(embedding_dim, hidden_dim, heads=4, concat=True),
            GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        ])
        
        # Policy network: outputs probability distribution over relations
        self.policy_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # question + path + current_entity
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_relations)
        )
        
        # Entity projection for path encoding
        self.entity_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Projection layers to ensure consistent hidden_dim output
        self.question_proj = nn.Linear(hidden_dim, hidden_dim)
        self.path_proj = nn.Linear(hidden_dim, hidden_dim)
        self.entity_proj_final = nn.Linear(hidden_dim, hidden_dim)
        
    def encode_question(self, question_emb):
        """Encode question into hidden representation"""
        if question_emb.dim() == 1:
            question_emb = question_emb.unsqueeze(0)  # Add batch dimension
        return self.question_encoder(question_emb)
    
    def encode_path(self, path_entities, path_relations_emb, device):
        """
        Encode current path into hidden representation
        
        Args:
            path_entities: List of entity embeddings in path
            path_relations_emb: List of relation embeddings in path (SBERT encoded)
            device: Device to put tensors on
        """
        if not path_entities or len(path_entities) <= 1:
            # No path yet, return zero encoding
            return torch.zeros(1, self.hidden_dim).to(device)
        
        # Prepare LSTM input: [entity, relation, entity, relation, ...]
        lstm_input = []
        for i in range(len(path_entities) - 1):
            entity_emb = path_entities[i].to(device)  # [embedding_dim]
            relation_emb = path_relations_emb[i].to(device)  # [embedding_dim]
            
            # Project entity embedding
            entity_emb = self.entity_proj(entity_emb)
            
            # Concatenate entity and relation embeddings
            step_input = torch.cat([entity_emb, relation_emb], dim=0)  # [embedding_dim * 2]
            lstm_input.append(step_input)
        
        if lstm_input:
            lstm_input = torch.stack(lstm_input).unsqueeze(0).to(device)  # [1, path_len, entity_dim + relation_dim]
            
            # LSTM encoding
            output, (hidden, _) = self.path_encoder(lstm_input)
            return hidden.squeeze(0)  # [1, hidden_dim]
        else:
            return torch.zeros(1, self.hidden_dim).to(device)
    
    def get_entity_context(self, entity_emb, neighbor_entities=None, device=None):
        """
        Get contextual representation of current entity using GAT
        
        Args:
            entity_emb: Current entity embedding
            neighbor_entities: List of neighbor entity embeddings (optional)
            device: Device to put tensors on
        """
        if neighbor_entities is None or not neighbor_entities:
            # No neighbors, apply GAT with self-loop
            if entity_emb.dim() == 1:
                entity_emb = entity_emb.unsqueeze(0)  # [1, embedding_dim]
            
            # Create self-loop edge
            edge_index = torch.LongTensor([[0], [0]]).to(device if device else entity_emb.device)
            
            # Apply GAT layers with self-loop
            x = entity_emb
            for gat_layer in self.gat_layers:
                x = F.relu(gat_layer(x, edge_index))
                x = F.dropout(x, p=0.3, training=self.training)
            
            return x  # [1, hidden_dim]
        
        # Create a mini-graph with entity and its neighbors
        node_features = [entity_emb.to(device)] + [ne.to(device) for ne in neighbor_entities]
        node_features = torch.stack(node_features)  # [num_nodes, embedding_dim]
        
        # Create edges (entity connects to all neighbors)
        num_neighbors = len(neighbor_entities)
        edge_index = []
        for i in range(1, num_neighbors + 1):
            edge_index.append([0, i])  # entity -> neighbor
            edge_index.append([i, 0])  # neighbor -> entity
        
        if edge_index:
            edge_index = torch.LongTensor(edge_index).transpose(0, 1).to(device)  # [2, num_edges]
        else:
            edge_index = torch.LongTensor([[0], [0]]).to(device)  # Self-loop
        
        # Apply GAT layers
        x = node_features
        for gat_layer in self.gat_layers:
            x = F.relu(gat_layer(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
        
        # Return representation of the central entity (index 0)
        return x[0].unsqueeze(0)  # [1, hidden_dim]
    
    def forward(self, state, device=None):
        """
        Forward pass: compute policy distribution over relations
        
        Args:
            state: Dict containing:
                - question_emb: Question embedding [embedding_dim]
                - current_entity_emb: Current entity embedding [embedding_dim]
                - path_entities: List of entity embeddings in current path
                - path_relations: List of relation embeddings in current path
                - neighbor_entities: List of neighbor entity embeddings (optional)
                - valid_relations: List of valid relation indices from current entity
            device: Device to put tensors on
        
        Returns:
            action_probs: Probability distribution over valid relations
            action_logits: Raw logits for valid relations
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Encode components
        question_emb = state['question_emb'].to(device)
        question_repr = self.encode_question(question_emb)  # [1, hidden_dim]
        
        path_repr = self.encode_path(
            state['path_entities'], 
            state['path_relations'], 
            device
        )  # [1, hidden_dim]
        
        current_entity_emb = state['current_entity_emb'].to(device)
        entity_repr = self.get_entity_context(
            current_entity_emb, 
            state.get('neighbor_entities'),
            device
        )  # [1, hidden_dim]
        
        # Apply final projections to ensure consistent dimensions
        question_repr = self.question_proj(question_repr)  # [1, hidden_dim]
        path_repr = self.path_proj(path_repr)              # [1, hidden_dim]
        entity_repr = self.entity_proj_final(entity_repr)  # [1, hidden_dim]
        
        # Combine representations
        combined_repr = torch.cat([
            question_repr, 
            path_repr, 
            entity_repr
        ], dim=1)  # [1, hidden_dim * 3]
        
        # Policy network
        all_relation_logits = self.policy_network(combined_repr)  # [1, num_relations]
        
        # Mask invalid relations
        valid_relations = state['valid_relations']
        if valid_relations:
            # Create mask for valid relations
            mask = torch.full((self.num_relations,), float('-inf')).to(device)
            for rel_idx in valid_relations:
                mask[rel_idx] = 0
            
            masked_logits = all_relation_logits.squeeze(0) + mask  # [num_relations]
            
            # Only keep logits for valid relations
            valid_logits = torch.stack([masked_logits[i] for i in valid_relations])  # [num_valid]
            action_probs = F.softmax(valid_logits, dim=0)  # [num_valid]
            
            return action_probs, valid_logits
        else:
            # No valid relations (shouldn't happen in well-formed KG)
            return torch.tensor([1.0]).to(device), torch.tensor([0.0]).to(device)

class GeneratorPretrainer:
    """Pretrainer for the generator using imitation learning (behavior cloning)"""
    
    def __init__(self, generator, dataset, embedding_manager, device='cuda'):
        self.generator = generator
        self.dataset = dataset
        self.embedding_manager = embedding_manager
        self.device = device
        
        self.generator.to(device)
        
        # Loss function: cross-entropy for supervised learning
        self.criterion = nn.CrossEntropyLoss()
        
    def prepare_training_state(self, question, head_entity, golden_path):
        """
        Prepare training state for imitation learning
        
        Args:
            question: Question text
            head_entity: Head entity name
            golden_path: Golden path dict with 'entities', 'relations'
            
        Returns:
            states: List of states for each step in the path
            targets: List of target relation indices for each step
        """
        # Use SBERT to encode question (clone to ensure requires_grad)
        question_emb = self.embedding_manager.encode_question(question).clone()
        
        # Prepare states and targets for each step
        states = []
        targets = []
        
        path_entities = golden_path['entities']  # [head_id, ..., answer_id]
        path_relations = golden_path['relations']  # [rel1, ...]
        
        # For each step in the golden path
        for step in range(len(path_relations)):
            current_entity_id = path_entities[step]
            target_relation = path_relations[step]
            
            # Get current entity embedding (clone to ensure requires_grad)
            current_entity_emb = self.embedding_manager.get_entity_embedding(current_entity_id).clone()
            
            # Get path so far (entity embeddings - clone to ensure requires_grad)
            path_entities_emb = [self.embedding_manager.get_entity_embedding(eid).clone() for eid in path_entities[:step+1]]
            
            # Get path relations so far (embeddings from SBERT - clone to ensure requires_grad)
            path_relations_emb = []
            for rel_name in path_relations[:step]:
                rel_emb = self.embedding_manager.get_relation_embedding(rel_name).clone()
                path_relations_emb.append(rel_emb)
            
            # Get valid relations from current entity
            valid_relation_names = self.dataset.get_valid_relations(current_entity_id)
            valid_relation_indices = []
            for rel_name in valid_relation_names:
                if rel_name in self.embedding_manager.relation_to_id:
                    valid_relation_indices.append(self.embedding_manager.relation_to_id[rel_name])
            
            # Get neighbor entities (for context)
            neighbor_entities_emb = []
            for rel_name, neighbor_id in self.dataset.entity_relations[current_entity_id]:
                if neighbor_id != current_entity_id:  # Avoid self-loops
                    neighbor_emb = self.embedding_manager.get_entity_embedding(neighbor_id).clone()
                    neighbor_entities_emb.append(neighbor_emb)
            
            # Create state
            state = {
                'question_emb': question_emb,
                'current_entity_emb': current_entity_emb,
                'path_entities': path_entities_emb,
                'path_relations': path_relations_emb,  # Now using SBERT embeddings
                'neighbor_entities': neighbor_entities_emb[:10],  # Limit neighbors
                'valid_relations': valid_relation_indices
            }
            
            # Target relation index
            if target_relation in self.embedding_manager.relation_to_id:
                target_idx = self.embedding_manager.relation_to_id[target_relation]
                # Find position of target in valid relations
                if target_idx in valid_relation_indices:
                    target_pos = valid_relation_indices.index(target_idx)
                    states.append(state)
                    targets.append(target_pos)  # Position in valid relations list
        
        return states, targets
    
    def train_epoch(self, dataloader, optimizer):
        """Train generator for one epoch using imitation learning"""
        self.generator.train()
        total_loss = 0
        total_accuracy = 0
        num_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            
            optimizer.zero_grad()
            batch_loss = 0
            batch_accuracy = 0
            batch_samples = 0
            
            # Process each sample in batch
            for i in range(len(batch['questions'])):
                question = batch['questions'][i]
                head_entity = batch['head_entities'][i]
                golden_paths = batch['golden_paths'][i]
                
                if not golden_paths:
                    continue
                
                # Use first golden path
                golden_path = golden_paths[0]
                
                try:
                    # Prepare training states
                    states, targets = self.prepare_training_state(question, head_entity, golden_path)
                    
                    if not states or not targets:
                        continue
                    
                    # Train on each state-target pair
                    for state, target in zip(states, targets):
                        # Move tensors to device
                        for key, value in state.items():
                            if isinstance(value, torch.Tensor):
                                state[key] = value.to(self.device)
                            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                                state[key] = [v.to(self.device) for v in value]
                        
                        # Forward pass
                        action_probs, action_logits = self.generator(state, self.device)
                        
                        if len(action_logits) > target:
                            # Compute loss
                            target_tensor = torch.LongTensor([target]).to(self.device)
                            loss = self.criterion(action_logits.unsqueeze(0), target_tensor)
                            
                            batch_loss += loss
                            
                            # Compute accuracy
                            predicted = torch.argmax(action_probs)
                            if predicted == target:
                                batch_accuracy += 1
                            
                            batch_samples += 1
                
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
            
            if batch_samples > 0:
                # Average loss for this batch
                batch_loss = batch_loss / batch_samples
                
                # Backward pass
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update metrics
                total_loss += batch_loss.item()
                total_accuracy += batch_accuracy
                num_samples += batch_samples
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}, Loss: {batch_loss.item():.4f}, "
                          f"Acc: {batch_accuracy/batch_samples:.4f}")
        
        avg_loss = total_loss / max(len(dataloader), 1)
        avg_accuracy = total_accuracy / max(num_samples, 1)
        
        return avg_loss, avg_accuracy

def test_generator():
    """Test generator implementation"""
    from data_loader import create_data_loaders
    
    print("Testing Generator...")
    
    # Load data
    datasets, loaders = create_data_loaders(hop="1hop", batch_size=4)
    train_dataset = datasets['train']
    train_loader = loaders['train']
    
    # Create generator
    embedding_dim = train_dataset.embedding_dim
    num_relations = len(train_dataset.relations)
    generator = PathGenerator(embedding_dim, num_relations)
    
    print(f"Generator: {embedding_dim}-dim embeddings, {num_relations} relations")
    
    # Test with one batch
    for batch in train_loader:
        if batch is not None:
            pretrainer = GeneratorPretrainer(generator, train_dataset, device='cpu')
            
            # Test state preparation
            question = batch['questions'][0]
            head_entity = batch['head_entities'][0]
            golden_paths = batch['golden_paths'][0]
            
            if golden_paths:
                golden_path = golden_paths[0]
                print(f"Sample question: {question}")
                print(f"Golden path: {golden_path}")
                
                states, targets = pretrainer.prepare_training_state(question, head_entity, golden_path)
                print(f"Created {len(states)} training states")
                
                if states:
                    # Test forward pass
                    state = states[0]
                    action_probs, action_logits = generator(state)
                    print(f"Action probs shape: {action_probs.shape}")
                    print(f"Valid relations: {len(state['valid_relations'])}")
                    print(f"Target: {targets[0]}")
            
            break
    
    print("Generator test completed!")

if __name__ == "__main__":
    test_generator()