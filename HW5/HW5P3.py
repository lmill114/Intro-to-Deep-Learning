import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import math

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Sample data loaded directly as a list of tuples
english_to_french = [
    ("I am cold", "J'ai froid"),
    ("You are tired", "Tu es fatigué"),
    ("He is hungry", "Il a faim"),
    ("She is happy", "Elle est heureuse"),
    ("We are friends", "Nous sommes amis"),
    ("They are students", "Ils sont étudiants"),
    ("The cat is sleeping", "Le chat dort"),
    ("The sun is shining", "Le soleil brille"),
    ("We love music", "Nous aimons la musique"),
    ("She speaks French fluently", "Elle parle français couramment"),
    ("He enjoys reading books", "Il aime lire des livres"),
    ("They play soccer every weekend", "Ils jouent au football chaque week-end"),
    ("The movie starts at 7 PM", "Le film commence à 19 heures"),
    ("She wears a red dress", "Elle porte une robe rouge"),
    ("We cook dinner together", "Nous cuisinons le dîner ensemble"),
    ("He drives a blue car", "Il conduit une voiture bleue"),
    ("They visit museums often", "Ils visitent souvent des musées"),
    ("The restaurant serves delicious food", "Le restaurant sert une délicieuse cuisine"),
    ("She studies mathematics at university", "Elle étudie les mathématiques à l'université"),
    ("We watch movies on Fridays", "Nous regardons des films le vendredi"),
    ("He listens to music while jogging", "Il écoute de la musique en faisant du jogging"),
    ("They travel around the world", "Ils voyagent autour du monde"),
    ("The book is on the table", "Le livre est sur la table"),
    ("She dances gracefully", "Elle danse avec grâce"),
    ("We celebrate birthdays with cake", "Nous célébrons les anniversaires avec un gâteau"),
    ("He works hard every day", "Il travaille dur tous les jours"),
    ("They speak different languages", "Ils parlent différentes langues"),
    ("The flowers bloom in spring", "Les fleurs fleurissent au printemps"),
    ("She writes poetry in her free time", "Elle écrit de la poésie pendant son temps libre"),
    ("We learn something new every day", "Nous apprenons quelque chose de nouveau chaque jour"),
    ("The dog barks loudly", "Le chien aboie bruyamment"),
    ("He sings beautifully", "Il chante magnifiquement"),
    ("They swim in the pool", "Ils nagent dans la piscine"),
    ("The birds chirp in the morning", "Les oiseaux gazouillent le matin"),
    ("She teaches English at school", "Elle enseigne l'anglais à l'école"),
    ("We eat breakfast together", "Nous prenons le petit déjeuner ensemble"),
    ("He paints landscapes", "Il peint des paysages"),
    ("They laugh at the joke", "Ils rient de la blague"),
    ("The clock ticks loudly", "L'horloge tic-tac bruyamment"),
    ("She runs in the park", "Elle court dans le parc"),
    ("We travel by train", "Nous voyageons en train"),
    ("He writes a letter", "Il écrit une lettre"),
    ("They read books at the library", "Ils lisent des livres à la bibliothèque"),
    ("The baby cries", "Le bébé pleure"),
    ("She studies hard for exams", "Elle étudie dur pour les examens"),
    ("We plant flowers in the garden", "Nous plantons des fleurs dans le jardin"),
    ("He fixes the car", "Il répare la voiture"),
    ("They drink coffee in the morning", "Ils boivent du café le matin"),
    ("The sun sets in the evening", "Le soleil se couche le soir"),
    ("She dances at the party", "Elle danse à la fête"),
    ("We play music at the concert", "Nous jouons de la musique au concert"),
    ("He cooks dinner for his family", "Il cuisine le dîner pour sa famille"),
    ("They study French grammar", "Ils étudient la grammaire française"),
    ("The rain falls gently", "La pluie tombe doucement"),
    ("She sings a song", "Elle chante une chanson"),
    ("We watch a movie together", "Nous regardons un film ensemble"),
    ("He sleeps deeply", "Il dort profondément"),
    ("They travel to Paris", "Ils voyagent à Paris"),
    ("The children play in the park", "Les enfants jouent dans le parc"),
    ("She walks along the beach", "Elle se promène le long de la plage"),
    ("We talk on the phone", "Nous parlons au téléphone"),
    ("He waits for the bus", "Il attend le bus"),
    ("They visit the Eiffel Tower", "Ils visitent la tour Eiffel"),
    ("The stars twinkle at night", "Les étoiles scintillent la nuit"),
    ("She dreams of flying", "Elle rêve de voler"),
    ("We work in the office", "Nous travaillons au bureau"),
    ("He studies history", "Il étudie l'histoire"),
    ("They listen to the radio", "Ils écoutent la radio"),
    ("The wind blows gently", "Le vent souffle doucement"),
    ("She swims in the ocean", "Elle nage dans l'océan"),
    ("We dance at the wedding", "Nous dansons au mariage"),
    ("He climbs the mountain", "Il gravit la montagne"),
    ("They hike in the forest", "Ils font de la randonnée dans la forêt"),
    ("The cat meows loudly", "Le chat miaule bruyamment"),
    ("She paints a picture", "Elle peint un tableau"),
    ("We build a sandcastle", "Nous construisons un château de sable"),
    ("He sings in the choir", "Il chante dans le chœur"),
    ("They ride bicycles", "Ils font du vélo"),
    ("The coffee is hot", "Le café est chaud"),
    ("She wears glasses", "Elle porte des lunettes"),
    ("We visit our grandparents", "Nous rendons visite à nos grands-parents"),
    ("He plays the guitar", "Il joue de la guitare"),
    ("They go shopping", "Ils font du shopping"),
    ("The teacher explains the lesson", "Le professeur explique la leçon"),
    ("She takes the train to work", "Elle prend le train pour aller au travail"),
    ("We bake cookies", "Nous faisons des biscuits"),
    ("He washes his hands", "Il se lave les mains"),
    ("They enjoy the sunset", "Ils apprécient le coucher du soleil"),
    ("The river flows calmly", "La rivière coule calmement"),
    ("She feeds the cat", "Elle nourrit le chat"),
    ("We visit the museum", "Nous visitons le musée"),
    ("He fixes his bicycle", "Il répare son vélo"),
    ("They paint the walls", "Ils peignent les murs"),
    ("The baby sleeps peacefully", "Le bébé dort paisiblement"),
    ("She ties her shoelaces", "Elle attache ses lacets"),
    ("We climb the stairs", "Nous montons les escaliers"),
    ("He shaves in the morning", "Il se rase le matin"),
    ("They set the table", "Ils mettent la table"),
    ("The airplane takes off", "L'avion décolle"),
    ("She waters the plants", "Elle arrose les plantes"),
    ("We practice yoga", "Nous pratiquons le yoga"),
    ("He turns off the light", "Il éteint la lumière"),
    ("They play video games", "Ils jouent aux jeux vidéo"),
    ("The soup smells delicious", "La soupe sent délicieusement bon"),
    ("She locks the door", "Elle ferme la porte à clé"),
    ("We enjoy a picnic", "Nous profitons d'un pique-nique"),
    ("He checks his email", "Il vérifie ses emails"),
    ("They go to the gym", "Ils vont à la salle de sport"),
    ("The moon shines brightly", "La lune brille intensément"),
    ("She catches the bus", "Elle attrape le bus"),
    ("We greet our neighbors", "Nous saluons nos voisins"),
    ("He combs his hair", "Il se peigne les cheveux"),
    ("They wave goodbye", "Ils font un signe d'adieu")
]

# Use the entire dataset
pairs = english_to_french

# Split the data into training (80%) and test (20%) sets
random.shuffle(pairs)
split_idx = int(len(pairs) * 0.8)
train_pairs = pairs[:split_idx]
test_pairs = pairs[split_idx:]

# IMPORTANT CHANGE: Train on 100% of the data while still having a 20% test set
train_pairs = pairs  # Use all pairs for training

print(f"Training pairs: {len(train_pairs)}")
print(f"Test pairs: {len(test_pairs)}")

# Create vocabulary
def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence.lower().split())
    
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

# Generate English and French vocabularies
eng_sentences = [pair[0] for pair in pairs]
fr_sentences = [pair[1] for pair in pairs]

eng_vocab = build_vocab(eng_sentences)
fr_vocab = build_vocab(fr_sentences)

# Create reverse mappings
idx_to_eng = {idx: word for word, idx in eng_vocab.items()}
idx_to_fr = {idx: word for word, idx in fr_vocab.items()}

print(f"English vocabulary size: {len(eng_vocab)}")
print(f"French vocabulary size: {len(fr_vocab)}")

# Create dataset class
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=20):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        
        # Convert words to indices
        src_indices = [self.src_vocab.get(word.lower(), self.src_vocab['<unk>']) 
                      for word in src_text.split()]
        tgt_indices = [self.tgt_vocab['<sos>']] + \
                     [self.tgt_vocab.get(word.lower(), self.tgt_vocab['<unk>']) 
                      for word in tgt_text.split()] + \
                     [self.tgt_vocab['<eos>']]
        
        # Pad sequences
        src_indices = src_indices + [self.src_vocab['<pad>']] * (self.max_len - len(src_indices))
        src_indices = src_indices[:self.max_len]
        
        tgt_indices = tgt_indices + [self.tgt_vocab['<pad>']] * (self.max_len + 2 - len(tgt_indices))
        tgt_indices = tgt_indices[:self.max_len + 2]  # +2 for <sos> and <eos>
        
        # Create attention mask (1 for real tokens, 0 for padding)
        src_mask = [1 if idx != self.src_vocab['<pad>'] else 0 for idx in src_indices]
        
        return {
            'src': torch.tensor(src_indices),
            'tgt': torch.tensor(tgt_indices[:-1]),  # Input to decoder (without <eos>)
            'tgt_y': torch.tensor(tgt_indices[1:]),  # Expected output (without <sos>)
            'src_mask': torch.tensor(src_mask),
            'src_text': src_text,
            'tgt_text': tgt_text
        }

# Create dataloaders
train_dataset = TranslationDataset(train_pairs, eng_vocab, fr_vocab)
test_dataset = TranslationDataset(test_pairs, eng_vocab, fr_vocab)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Transformer model components
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, n_heads=2, num_layers=2, 
                 d_ff=512, max_len=100, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                                  dim_feedforward=d_ff, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, 
                                                  dim_feedforward=d_ff, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Source embedding with positional encoding
        src_embedded = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        
        # Create masks if not provided
        if src_mask is None:
            src_mask = (src != 0)
        
        if tgt_mask is None:
            tgt_seq_len = tgt.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        
        # Target embedding with positional encoding
        tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Transformer encoder
        # Key padding mask expects shape (batch_size, seq_len) with True for positions to mask
        key_padding_mask = ~src_mask.bool()  # Invert because PyTorch expects True for padding positions
        
        memory = self.transformer_encoder(src_embedded.transpose(0, 1), 
                                        src_key_padding_mask=key_padding_mask)
        
        # Transformer decoder
        output = self.transformer_decoder(tgt_embedded.transpose(0, 1), memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        
        return self.output_layer(output)

def train_model(model, dataloader, optimizer, criterion, n_epochs, device, clip=1.0):
    model.train()
    model.to(device)
    
    epoch_losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            tgt_y = batch['tgt_y'].to(device)
            src_mask = batch['src_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with transformer model
            output = model(src, tgt, src_mask)
            
            # Calculate loss
            output = output.contiguous().view(-1, output.shape[-1])
            tgt_y = tgt_y.contiguous().view(-1)
            
            loss = criterion(output, tgt_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss = epoch_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        
        print(f'Epoch: {epoch+1}, Loss: {epoch_loss:.4f}')
    
    return epoch_losses

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            tgt_y = batch['tgt_y'].to(device)
            src_mask = batch['src_mask'].to(device)
            
            # Forward pass with transformer model
            output = model(src, tgt, src_mask)
            
            # Calculate loss
            output_flat = output.contiguous().view(-1, output.shape[-1])
            tgt_flat = tgt_y.contiguous().view(-1)
            
            loss = criterion(output_flat, tgt_flat)
            total_loss += loss.item()
            
            # Calculate accuracy (ignoring padding tokens)
            preds = output.argmax(dim=2)
            non_pad = tgt_y != 0  # Ignore padding tokens
            correct = (preds == tgt_y) * non_pad
            
            total_correct += correct.sum().item()
            total_tokens += non_pad.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, accuracy

# Translation function
def translate_sentence(model, sentence, src_vocab, tgt_vocab, device, max_len=20):
    model.eval()
    
    # Tokenize and convert to indices
    tokens = sentence.lower().split()
    src_indices = [src_vocab.get(token, src_vocab['<unk>']) for token in tokens]
    
    # Pad if necessary
    src_indices = src_indices + [src_vocab['<pad>']] * (max_len - len(src_indices))
    src_indices = src_indices[:max_len]
    
    src_tensor = torch.tensor([src_indices]).to(device)
    src_mask = torch.tensor([[1 if idx != src_vocab['<pad>'] else 0 for idx in src_indices]]).to(device)
    
    # Start with <sos> token
    tgt_indices = [tgt_vocab['<sos>']]
    tgt_tensor = torch.tensor([tgt_indices]).to(device)
    
    for _ in range(max_len):
        # Make prediction
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, src_mask)
        
        # Get the next word prediction
        pred_token = output[0, -1].argmax().item()
        
        # Add to target sequence
        tgt_indices.append(pred_token)
        tgt_tensor = torch.tensor([tgt_indices]).to(device)
        
        # Stop if we predict <eos>
        if pred_token == tgt_vocab['<eos>']:
            break
    
    # Convert indices back to words (excluding <sos> and <eos>)
    idx_to_tgt = {idx: word for word, idx in tgt_vocab.items()}
    translated_tokens = [idx_to_tgt[idx] for idx in tgt_indices[1:-1] if idx in idx_to_tgt]  # Skip <sos> and <eos>
    
    return ' '.join(translated_tokens)

# Function to run grid search on transformer configurations
def run_grid_search():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = []
    
    for n_layers in [1, 2, 4]:
        for n_heads in [2, 4]:
            print(f"\nTraining model with {n_layers} layers and {n_heads} heads")
            
            # Initialize model
            model = TransformerModel(
                src_vocab_size=len(eng_vocab),
                tgt_vocab_size=len(fr_vocab),
                n_heads=n_heads,
                num_layers=n_layers
            )
            
            # Define optimizer and criterion
            optimizer = optim.Adam(model.parameters(), lr=0.0005)
            criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
            
            # Train the model
            train_losses = train_model(model, train_dataloader, optimizer, criterion, n_epochs=50, device=device)
            
            # Evaluate the model
            test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion, device)
            
            # Save results
            results.append({
                'model_type': 'Transformer',
                'n_layers': n_layers,
                'n_heads': n_heads,
                'train_loss': train_losses[-1],
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            })
            
            print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
            
            # Test some translations
            test_sentences = [
                "I am cold",
                "She speaks French fluently",
                "They visit the Eiffel Tower"
            ]
            
            print("\nSample Translations:")
            for sent in test_sentences:
                translated = translate_sentence(model, sent, eng_vocab, fr_vocab, device)
                print(f"English: {sent}")
                print(f"French (predicted): {translated}")
                
                # Find the ground truth if available
                ground_truth = None
                for pair in pairs:
                    if pair[0] == sent:
                        ground_truth = pair[1]
                        break
                
                if ground_truth:
                    print(f"French (actual): {ground_truth}")
                print()
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Extract and organize results for visualization
    transformer_results = results
    
    # Sort transformer results by accuracy
    transformer_results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    # Plot transformer results
    labels = [f"{r['n_layers']}L-{r['n_heads']}H" for r in transformer_results]
    accuracies = [r['test_accuracy'] * 100 for r in transformer_results]
    
    plt.bar(labels, accuracies, color='skyblue')
    
    plt.xlabel('Model Configuration')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Transformer Model Performance Comparison')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Create comparison table
    print("\nResults summary:")
    print("-" * 80)
    print(f"{'Model':<25} {'Train Loss':<12} {'Test Loss':<12} {'Accuracy':<10}")
    print("-" * 80)
    
    # Sort by accuracy
    all_results = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
    
    for r in all_results:
        model_name = f"Transformer {r['n_layers']}L-{r['n_heads']}H"
        print(f"{model_name:<25} {r['train_loss']:<12.4f} {r['test_loss']:<12.4f} {r['test_accuracy']:<10.4f}")
    
    return results

# Function to run a single model with best configuration
def run_best_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model with the best configuration
    model = TransformerModel(
        src_vocab_size=len(eng_vocab),
        tgt_vocab_size=len(fr_vocab),
        n_heads=4,
        num_layers=4,
        d_model=256,
        d_ff=1024
    )
    
    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    # Train the model
    train_losses = train_model(model, train_dataloader, optimizer, criterion, n_epochs=50, device=device)
    
    # Evaluate the model
    test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    
    # Test some translations
    test_sentences = [
        "I am cold",
        "She speaks French fluently",
        "They visit the Eiffel Tower",
        "We learn something new every day",
        "The sun sets in the evening",
        "The teacher explains the lesson"
    ]
    
    print("\nSample Translations:")
    for sent in test_sentences:
        translated = translate_sentence(model, sent, eng_vocab, fr_vocab, device)
        print(f"English: {sent}")
        print(f"French (predicted): {translated}")
        
        # Find the ground truth if available
        ground_truth = None
        for pair in pairs:
            if pair[0] == sent:
                ground_truth = pair[1]
                break
        
        if ground_truth:
            print(f"French (actual): {ground_truth}")
        print()
    
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()

if __name__ == "__main__":
    # Run grid search to find the best model configuration
    results = run_grid_search()
    
    # Uncomment to run the best model
    # run_best_model()