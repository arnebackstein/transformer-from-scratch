import torch
import torch.nn as nn
import torch.optim as optim
from transformer import SimpleTransformer
import os

def generate_data(batch_size, seq_len, vocab_size):
    """
    Generate sequences where each number is followed by its double (mod vocab_size)
    """
    sequences = torch.zeros((batch_size, seq_len + 1), dtype=torch.long)
    
    for b in range(batch_size):
        current = torch.randint(1, vocab_size//2, (1,))
        sequences[b, 0] = current
        
        for i in range(1, seq_len + 1):
            current = (current * 2) % vocab_size
            sequences[b, i] = current
    
    x = sequences[:, :-1]
    y = sequences[:, 1:]
    return x, y

def train():
    vocab_size = 37
    seq_len = 16
    batch_size = 32
    num_epochs = 1000
    learning_rate = 0.001
    weight_decay = 0.01
    
    model = SimpleTransformer(
        vocab_size=vocab_size,
        max_seq_len=seq_len
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
        x, y = generate_data(batch_size, seq_len, vocab_size)
        
        optimizer.zero_grad()
        logits = model(x)
        
        logits = logits.view(-1, vocab_size)
        y = y.reshape(-1)
        
        loss = criterion(logits, y)
        
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == y).float().mean()
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        accuracies.append(accuracy.item())
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
            
            with torch.no_grad():
                x = torch.tensor([[1]])
                print(f'Start: {x[0].tolist()}')
                
                for _ in range(8):
                    output = model(x)
                    next_token = torch.argmax(output[:, -1:], dim=-1)
                    x = torch.cat([x, next_token], dim=1)
                
                print(f'Generated: {x[0].tolist()}')
                print('---')
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    filename = f'models/transformer_model_{vocab_size}_{seq_len}_{batch_size}_{num_epochs}_{learning_rate}_{weight_decay}.pth'
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

if __name__ == '__main__':
    train() 