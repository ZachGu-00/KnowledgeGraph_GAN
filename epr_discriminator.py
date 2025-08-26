#!/usr/bin/env python3
"""
EPR-based Discriminator: Entity Reasoning Pattern + GATv2
Based on proposal.md section "Important: Pretraining for discriminator using EPR"

Architecture:
1. EPR Pre-training:
   - Bi-Encoder for AP (Atomic Pattern) recall 
   - Cross-Encoder for EP (Execution Pattern) ranking
2. GAT with EPR prior injection for final discrimination
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from transformers import BertModel, BertTokenizer
import numpy as np
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Optional
import json
import pickle

class BiEncoderEPR(nn.Module):
    """
    Bi-Encoder for Atomic Pattern (AP) recall
    Encodes questions and patterns separately for semantic alignment
    """
    
    def __init__(self, model_name='bert-base-uncased', hidden_dim=768):
        super(BiEncoderEPR, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        # Projection heads for question and pattern embeddings
        self.question_proj = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.pattern_proj = nn.Sequential(
            nn.Linear(768, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Temperature for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def encode_text(self, texts: List[str], max_length=64):
        """Encode texts using BERT"""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length,
            return_tensors='pt'
        ).to(self.bert.device)
        
        outputs = self.bert(**inputs)
        # Use [CLS] token representation
        return outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
    
    def forward(self, questions: List[str], patterns: List[str]):
        """
        Forward pass for contrastive learning
        
        Args:
            questions: List of question strings
            patterns: List of pattern strings (e.g., "movie -> director")
        
        Returns:
            Similarity scores for contrastive loss
        """
        # Encode questions and patterns
        q_emb = self.encode_text(questions)  # [batch_size, 768]
        p_emb = self.encode_text(patterns)   # [batch_size, 768]
        
        # Project to common space
        q_proj = F.normalize(self.question_proj(q_emb), p=2, dim=1)  # [batch_size, hidden_dim]
        p_proj = F.normalize(self.pattern_proj(p_emb), p=2, dim=1)   # [batch_size, hidden_dim]
        
        # Compute similarity matrix
        similarity = torch.matmul(q_proj, p_proj.T) / self.temperature  # [batch_size, batch_size]
        
        return {
            'similarity': similarity,
            'q_embeddings': q_proj,
            'p_embeddings': p_proj
        }
    
    def get_pattern_scores(self, question: str, patterns: List[str], top_k=30):
        """
        Get top-k most relevant patterns for a question
        
        Args:
            question: Single question string
            patterns: List of candidate patterns
            top_k: Number of top patterns to return
            
        Returns:
            Top-k patterns with scores
        """
        with torch.no_grad():
            q_emb = self.encode_text([question])
            p_emb = self.encode_text(patterns)
            
            q_proj = F.normalize(self.question_proj(q_emb), p=2, dim=1)
            p_proj = F.normalize(self.pattern_proj(p_emb), p=2, dim=1)
            
            scores = torch.matmul(q_proj, p_proj.T).squeeze(0)  # [num_patterns]
            top_scores, top_indices = torch.topk(scores, min(top_k, len(patterns)))
            
            return [(patterns[i], top_scores[j].item()) for j, i in enumerate(top_indices)]

class CrossEncoderEPR(nn.Module):
    """
    Cross-Encoder for Execution Pattern (EP) ranking
    Jointly encodes question + execution pattern for logical coherence
    """
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2):
        super(CrossEncoderEPR, self).__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        # Classification head for EP scoring
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)  # [irrelevant, relevant]
        )
        
    def forward(self, questions: List[str], execution_patterns: List[str]):
        """
        Forward pass for EP ranking
        
        Args:
            questions: List of question strings
            execution_patterns: List of EP strings (e.g., "movie -> director -> ?x")
        
        Returns:
            Classification scores
        """
        # Concatenate question + [SEP] + execution pattern
        combined_inputs = [
            f"{q} [SEP] {ep}" for q, ep in zip(questions, execution_patterns)
        ]
        
        inputs = self.tokenizer(
            combined_inputs,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.bert.device)
        
        outputs = self.bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        logits = self.classifier(cls_embedding)  # [batch_size, num_classes]
        
        return logits
    
    def get_ep_scores(self, question: str, execution_patterns: List[str], top_k=10):
        """
        Get top-k execution patterns for a question
        
        Args:
            question: Single question string
            execution_patterns: List of candidate EPs
            top_k: Number of top EPs to return
            
        Returns:
            Top-k EPs with relevance scores
        """
        with torch.no_grad():
            questions = [question] * len(execution_patterns)
            logits = self.forward(questions, execution_patterns)
            
            # Get relevance scores (softmax on class 1)
            scores = F.softmax(logits, dim=1)[:, 1]  # [num_eps]
            top_scores, top_indices = torch.topk(scores, min(top_k, len(execution_patterns)))
            
            return [(execution_patterns[i], top_scores[j].item()) for j, i in enumerate(top_indices)]

class EPRInformedGAT(nn.Module):
    """
    GAT with EPR prior injection
    Uses EPR scores to initialize attention weights and guide reasoning
    """
    
    def __init__(self, 
                 node_dim=384,  # SBERT embedding dimension
                 edge_dim=384,  # Relation embedding dimension  
                 hidden_dim=128,
                 num_heads=4,
                 num_layers=3,
                 dropout=0.2):
        super(EPRInformedGAT, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Question encoder for conditioning
        self.question_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # EPR prior injection layer
        self.epr_injection = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),  # EPR score -> feature
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_heads)  # -> attention heads
        )
        
        # GAT layers with EPR conditioning
        self.gat_layers = nn.ModuleList()
        
        # First layer: node_dim -> hidden_dim
        self.gat_layers.append(
            GATv2Conv(
                node_dim, 
                hidden_dim // num_heads,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim + num_heads  # relation + EPR features
            )
        )
        
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim + num_heads
                )
            )
        
        # Final layer: hidden_dim -> hidden_dim (average heads)
        self.gat_layers.append(
            GATv2Conv(
                hidden_dim,
                hidden_dim,
                heads=num_heads,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim + num_heads
            )
        )
        
        # Path scoring head
        self.path_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                node_features: torch.Tensor,    # [num_nodes, node_dim]
                edge_index: torch.Tensor,       # [2, num_edges] 
                edge_attr: torch.Tensor,        # [num_edges, edge_dim]
                epr_scores: torch.Tensor,       # [num_edges] EPR scores for each edge
                question_emb: torch.Tensor,     # [node_dim] question embedding
                path_mask: Optional[torch.Tensor] = None  # [num_nodes] mask for reasoning path
                ):
        """
        Forward pass with EPR prior injection
        
        Args:
            node_features: Entity embeddings
            edge_index: Graph connectivity 
            edge_attr: Relation embeddings
            epr_scores: EPR relevance scores for each edge
            question_emb: Question embedding for conditioning
            path_mask: Boolean mask indicating reasoning path nodes
            
        Returns:
            Path relevance score
        """
        batch_size = question_emb.size(0) if question_emb.dim() > 1 else 1
        
        # Encode question for conditioning
        q_encoded = self.question_encoder(question_emb)  # [hidden_dim]
        if q_encoded.dim() == 1:
            q_encoded = q_encoded.unsqueeze(0)  # [1, hidden_dim]
        
        # Inject EPR prior into edge features
        epr_features = self.epr_injection(epr_scores.unsqueeze(-1))  # [num_edges, num_heads]
        enhanced_edge_attr = torch.cat([edge_attr, epr_features], dim=-1)  # [num_edges, edge_dim + num_heads]
        
        # Apply GAT layers with EPR conditioning
        x = node_features
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index, edge_attr=enhanced_edge_attr)
            if i < len(self.gat_layers) - 1:  # Apply activation except for last layer
                x = F.elu(x)
        
        # Focus on reasoning path nodes if mask provided
        if path_mask is not None:
            # Weighted pooling with focus on path nodes
            path_nodes = x[path_mask]  # [num_path_nodes, hidden_dim]
            context_nodes = x[~path_mask]  # [num_context_nodes, hidden_dim]
            
            # Give higher weight to path nodes
            path_weight = 0.8
            context_weight = 0.2
            
            if len(path_nodes) > 0 and len(context_nodes) > 0:
                pooled_path = torch.mean(path_nodes, dim=0)  # [hidden_dim]
                pooled_context = torch.mean(context_nodes, dim=0)  # [hidden_dim]
                graph_repr = path_weight * pooled_path + context_weight * pooled_context
            elif len(path_nodes) > 0:
                graph_repr = torch.mean(path_nodes, dim=0)
            else:
                graph_repr = torch.mean(x, dim=0)  # [hidden_dim]
        else:
            # Simple mean pooling
            graph_repr = torch.mean(x, dim=0)  # [hidden_dim]
        
        # Compute path relevance score
        path_score = self.path_scorer(graph_repr)  # [1]
        
        return {
            'path_score': path_score,
            'node_representations': x,
            'graph_representation': graph_repr
        }

class EPRDiscriminator(nn.Module):
    """
    Complete EPR-based Discriminator combining all components
    """
    
    def __init__(self, 
                 node_dim=384,
                 edge_dim=384,
                 hidden_dim=128,
                 device='cuda'):
        super(EPRDiscriminator, self).__init__()
        
        self.device = device
        
        # EPR components
        self.bi_encoder = BiEncoderEPR(hidden_dim=hidden_dim)
        self.cross_encoder = CrossEncoderEPR()
        self.gat_discriminator = EPRInformedGAT(
            node_dim=node_dim,
            edge_dim=edge_dim, 
            hidden_dim=hidden_dim
        )
        
        # Move to device
        self.to(device)
        
    def extract_patterns(self, kb_graph, max_patterns=1000):
        """
        Extract Entity-Relation Atomic Patterns (ER-APs) and 
        Relation-Relation Atomic Patterns (RR-APs) from KB
        """
        er_patterns = set()  # Entity-Relation patterns
        rr_patterns = set()  # Relation-Relation patterns
        
        # Extract ER-APs: entity_type -> relation
        for source, target, data in kb_graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            # Simplified: use entity as entity_type  
            er_patterns.add(f"{source} -> {relation}")
            er_patterns.add(f"{relation} -> {target}")
        
        # Extract RR-APs: relation1 <-> relation2 (connected via shared entities)
        relation_connections = defaultdict(set)
        for source, target, data in kb_graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            # Find relations connected to same entities
            for _, _, data2 in kb_graph.edges(source, data=True):
                rel2 = data2.get('relation', 'unknown')
                if rel2 != relation:
                    rr_patterns.add(f"{relation} <-> {rel2}")
        
        all_patterns = list(er_patterns) + list(rr_patterns)
        return all_patterns[:max_patterns]
    
    def generate_execution_patterns(self, atomic_patterns, max_length=3, max_eps=500):
        """
        Generate Execution Patterns (EPs) by combining Atomic Patterns
        """
        execution_patterns = set()
        
        # 1-hop EPs from ER-APs
        for pattern in atomic_patterns:
            if ' -> ' in pattern and ' <-> ' not in pattern:
                execution_patterns.add(f"{pattern} -> ?x")
        
        # 2-hop EPs by chaining
        er_patterns = [p for p in atomic_patterns if ' -> ' in p and ' <-> ' not in p]
        for i, p1 in enumerate(er_patterns[:100]):  # Limit for efficiency
            for p2 in er_patterns[i+1:100]:
                # Try to chain if p1 target matches p2 source
                if p1.split(' -> ')[1] == p2.split(' -> ')[0]:
                    ep = f"{p1.split(' -> ')[0]} -> {p1.split(' -> ')[1]} -> {p2.split(' -> ')[1]} -> ?x"
                    execution_patterns.add(ep)
        
        return list(execution_patterns)[:max_eps]
    
    def forward(self,
                question: str,
                node_features: torch.Tensor,
                edge_index: torch.Tensor, 
                edge_attr: torch.Tensor,
                path_mask: torch.Tensor,
                atomic_patterns: List[str],
                execution_patterns: List[str]):
        """
        Complete forward pass through EPR pipeline
        
        1. Use Bi-Encoder to get relevant APs
        2. Use Cross-Encoder to get relevant EPs  
        3. Use GAT with EPR priors for final discrimination
        """
        # Step 1: Get relevant atomic patterns using Bi-Encoder
        relevant_aps = self.bi_encoder.get_pattern_scores(question, atomic_patterns, top_k=30)
        ap_scores = torch.tensor([score for _, score in relevant_aps]).to(self.device)
        
        # Step 2: Get relevant execution patterns using Cross-Encoder
        relevant_eps = self.cross_encoder.get_ep_scores(question, execution_patterns, top_k=10)
        ep_scores = torch.tensor([score for _, score in relevant_eps]).to(self.device)
        
        # Step 3: Combine scores as EPR prior (simple average for now)
        # Map edge scores based on pattern relevance (simplified mapping)
        num_edges = edge_index.size(1)
        epr_scores = torch.ones(num_edges).to(self.device) * 0.5  # Default score
        
        # Enhanced: could map specific edges to pattern scores based on relation types
        
        # Step 4: Apply GAT with EPR injection
        question_emb = self.bi_encoder.encode_text([question])[0]  # [768] -> [node_dim]
        
        gat_output = self.gat_discriminator(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            epr_scores=epr_scores,
            question_emb=question_emb,
            path_mask=path_mask
        )
        
        return {
            'path_score': gat_output['path_score'],
            'relevant_aps': relevant_aps,
            'relevant_eps': relevant_eps,
            'epr_scores': epr_scores,
            'node_representations': gat_output['node_representations']
        }


def test_epr_discriminator():
    """Test EPR discriminator implementation"""
    print("Testing EPR Discriminator...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create EPR discriminator
    discriminator = EPRDiscriminator(device=device)
    
    # Mock data for testing
    question = "Who directed The Godfather?"
    node_features = torch.randn(10, 384).to(device)  # 10 nodes, SBERT dim
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]).to(device)  # Simple chain
    edge_attr = torch.randn(4, 384).to(device)  # 4 edges, relation embeddings
    path_mask = torch.tensor([True, True, False, False, False, False, False, False, False, False]).to(device)
    
    # Mock patterns
    atomic_patterns = [
        "movie -> director",
        "person -> film",
        "director <-> actor",
        "movie -> genre"
    ]
    
    execution_patterns = [
        "movie -> director -> ?x",
        "person -> film -> genre -> ?x",
        "movie -> actor -> film -> ?x"
    ]
    
    # Test forward pass
    try:
        output = discriminator(
            question=question,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            path_mask=path_mask,
            atomic_patterns=atomic_patterns,
            execution_patterns=execution_patterns
        )
        
        print(f"✓ Forward pass successful!")
        print(f"  Path score: {output['path_score'].item():.4f}")
        print(f"  Relevant APs: {len(output['relevant_aps'])}")
        print(f"  Relevant EPs: {len(output['relevant_eps'])}")
        print(f"  Top AP: {output['relevant_aps'][0] if output['relevant_aps'] else 'None'}")
        print(f"  Top EP: {output['relevant_eps'][0] if output['relevant_eps'] else 'None'}")
        
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_epr_discriminator()