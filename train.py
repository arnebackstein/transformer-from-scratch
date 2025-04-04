import torch
import torch.nn as nn
import torch.optim as optim
from transformer import SimpleTransformer

def generate_data(batch_size, seq_len, vocab_size):
    """
    Generate sequences where each number is followed by its double (mod vocab_size)
    Example with vocab_size=10: [1,2,4,8,6,2,4,8,6,2] (doubling, but mod vocab_size)
    """
    # Initialize sequences
    sequences = torch.zeros((batch_size, seq_len + 1), dtype=torch.long)
    
    # For each sequence in the batch
    for b in range(batch_size):
        # Start with a random number between 1 and vocab_size//2
        current = torch.randint(1, vocab_size//2, (1,))
        sequences[b, 0] = current
        
        # Generate the rest of the sequence
        for i in range(1, seq_len + 1):
            # Double the previous number, use modulo to stay within vocab_size
            current = (current * 2) % vocab_size
            sequences[b, i] = current
    
    # Input: all tokens except last
    x = sequences[:, :-1]
    # Target: all tokens except first (what comes next)
    y = sequences[:, 1:]
    return x, y

def train():
    # Hyperparameters
    vocab_size = 3447     # Increased for more interesting patterns
    seq_len = 16        # Length of input sequence
    batch_size = 32    # Increased batch size
    num_epochs = 1000  # More epochs to observe grokking
    learning_rate = 0.001
    weight_decay = 0.01 # Added weight decay for regularization
    
    # Initialize model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        max_seq_len=seq_len
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(  # Changed to AdamW
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay  # L2 regularization
    )
    
    # Track training metrics
    losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
        # Get batch of sequences
        x, y = generate_data(batch_size, seq_len, vocab_size)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(x)
        
        # Reshape for loss calculation
        logits = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
        y = y.reshape(-1)                        # (batch_size * seq_len)
        
        # Calculate loss
        loss = criterion(logits, y)
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == y).float().mean()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Store metrics
        losses.append(loss.item())
        accuracies.append(accuracy.item())
        
        # Print progress and test
        if (epoch + 1) % 50 == 0:  # Changed to print more frequently
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
            
            # Test generation
            with torch.no_grad():
                x = torch.tensor([[1]])
                print(f'Start: {x[0].tolist()}')
                
                # Generate more tokens to see pattern
                for _ in range(8):  # Increased to 8 tokens
                    output = model(x)
                    next_token = torch.argmax(output[:, -1:], dim=-1)
                    x = torch.cat([x, next_token], dim=1)
                
                print(f'Generated: {x[0].tolist()}')
                print('---')

if __name__ == '__main__':
    train() 