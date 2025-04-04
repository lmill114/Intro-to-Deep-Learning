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
# Reverse the pairs to translate from French to English
french_to_english = [
    ("J'ai froid", "I am cold"),
    ("Tu es fatigué", "You are tired"),
    ("Il a faim", "He is hungry"),
    ("Elle est heureuse", "She is happy"),
    ("Nous sommes amis", "We are friends"),
    ("Ils sont étudiants", "They are students"),
    ("Le chat dort", "The cat is sleeping"),
    ("Le soleil brille", "The sun is shining"),
    ("Nous aimons la musique", "We love music"),
    ("Elle parle français couramment", "She speaks French fluently"),
    ("Il aime lire des livres", "He enjoys reading books"),
    ("Ils jouent au football chaque week-end", "They play soccer every weekend"),
    ("Le film commence à 19 heures", "The movie starts at 7 PM"),
    ("Elle porte une robe rouge", "She wears a red dress"),
    ("Nous cuisinons le dîner ensemble", "We cook dinner together"),
    ("Il conduit une voiture bleue", "He drives a blue car"),
    ("Ils visitent souvent des musées", "They visit museums often"),
    ("Le restaurant sert une délicieuse cuisine", "The restaurant serves delicious food"),
    ("Elle étudie les mathématiques à l'université", "She studies mathematics at university"),
    ("Nous regardons des films le vendredi", "We watch movies on Fridays"),
    ("Il écoute de la musique en faisant du jogging", "He listens to music while jogging"),
    ("Ils voyagent autour du monde", "They travel around the world"),
    ("Le livre est sur la table", "The book is on the table"),
    ("Elle danse avec grâce", "She dances gracefully"),
    ("Nous célébrons les anniversaires avec un gâteau", "We celebrate birthdays with cake"),
    ("Il travaille dur tous les jours", "He works hard every day"),
    ("Ils parlent différentes langues", "They speak different languages"),
    ("Les fleurs fleurissent au printemps", "The flowers bloom in spring"),
    ("Elle écrit de la poésie pendant son temps libre", "She writes poetry in her free time"),
    ("Nous apprenons quelque chose de nouveau chaque jour", "We learn something new every day"),
    ("Le chien aboie bruyamment", "The dog barks loudly"),
    ("Il chante magnifiquement", "He sings beautifully"),
    ("Ils nagent dans la piscine", "They swim in the pool"),
    ("Les oiseaux gazouillent le matin", "The birds chirp in the morning"),
    ("Elle enseigne l'anglais à l'école", "She teaches English at school"),
    ("Nous prenons le petit déjeuner ensemble", "We eat breakfast together"),
    ("Il peint des paysages", "He paints landscapes"),
    ("Ils rient de la blague", "They laugh at the joke"),
    ("L'horloge tic-tac bruyamment", "The clock ticks loudly"),
    ("Elle court dans le parc", "She runs in the park"),
    ("Nous voyageons en train", "We travel by train"),
    ("Il écrit une lettre", "He writes a letter"),
    ("Ils lisent des livres à la bibliothèque", "They read books at the library"),
    ("Le bébé pleure", "The baby cries"),
    ("Elle étudie dur pour les examens", "She studies hard for exams"),
    ("Nous plantons des fleurs dans le jardin", "We plant flowers in the garden"),
    ("Il répare la voiture", "He fixes the car"),
    ("Ils boivent du café le matin", "They drink coffee in the morning"),
    ("Le soleil se couche le soir", "The sun sets in the evening"),
    ("Elle danse à la fête", "She dances at the party"),
    ("Nous jouons de la musique au concert", "We play music at the concert"),
    ("Il cuisine le dîner pour sa famille", "He cooks dinner for his family"),
    ("Ils étudient la grammaire française", "They study French grammar"),
    ("La pluie tombe doucement", "The rain falls gently"),
    ("Elle chante une chanson", "She sings a song"),
    ("Nous regardons un film ensemble", "We watch a movie together"),
    ("Il dort profondément", "He sleeps deeply"),
    ("Ils voyagent à Paris", "They travel to Paris"),
    ("Les enfants jouent dans le parc", "The children play in the park"),
    ("Elle se promène le long de la plage", "She walks along the beach"),
    ("Nous parlons au téléphone", "We talk on the phone"),
    ("Il attend le bus", "He waits for the bus"),
    ("Ils visitent la tour Eiffel", "They visit the Eiffel Tower"),
    ("Les étoiles scintillent la nuit", "The stars twinkle at night"),
    ("Elle rêve de voler", "She dreams of flying"),
    ("Nous travaillons au bureau", "We work in the office"),
    ("Il étudie l'histoire", "He studies history"),
    ("Ils écoutent la radio", "They listen to the radio"),
    ("Le vent souffle doucement", "The wind blows gently"),
    ("Elle nage dans l'océan", "She swims in the ocean"),
    ("Nous dansons au mariage", "We dance at the wedding"),
    ("Il gravit la montagne", "He climbs the mountain"),
    ("Ils font de la randonnée dans la forêt", "They hike in the forest"),
    ("Le chat miaule bruyamment", "The cat meows loudly"),
    ("Elle peint un tableau", "She paints a picture"),
    ("Nous construisons un château de sable", "We build a sandcastle"),
    ("Il chante dans le chœur", "He sings in the choir"),
    ("Ils font du vélo", "They ride bicycles"),
    ("Le café est chaud", "The coffee is hot"),
    ("Elle porte des lunettes", "She wears glasses"),
    ("Nous rendons visite à nos grands-parents", "We visit our grandparents"),
    ("Il joue de la guitare", "He plays the guitar"),
    ("Ils font du shopping", "They go shopping"),
    ("Le professeur explique la leçon", "The teacher explains the lesson"),
    ("Elle prend le train pour aller au travail", "She takes the train to work"),
    ("Nous faisons des biscuits", "We bake cookies"),
    ("Il se lave les mains", "He washes his hands"),
    ("Ils apprécient le coucher du soleil", "They enjoy the sunset"),
    ("La rivière coule calmement", "The river flows calmly"),
    ("Elle nourrit le chat", "She feeds the cat"),
    ("Nous visitons le musée", "We visit the museum"),
    ("Il répare son vélo", "He fixes his bicycle"),
    ("Ils peignent les murs", "They paint the walls"),
    ("Le bébé dort paisiblement", "The baby sleeps peacefully"),
    ("Elle attache ses lacets", "She ties her shoelaces"),
    ("Nous montons les escaliers", "We climb the stairs"),
    ("Il se rase le matin", "He shaves in the morning"),
    ("Ils mettent la table", "They set the table"),
    ("L'avion décolle", "The airplane takes off"),
    ("Elle arrose les plantes", "She waters the plants"),
    ("Nous pratiquons le yoga", "We practice yoga"),
    ("Il éteint la lumière", "He turns off the light"),
    ("Ils jouent aux jeux vidéo", "They play video games"),
    ("La soupe sent délicieusement bon", "The soup smells delicious"),
    ("Elle ferme la porte à clé", "She locks the door"),
    ("Nous profitons d'un pique-nique", "We enjoy a picnic"),
    ("Il vérifie ses emails", "He checks his email"),
    ("Ils vont à la salle de sport", "They go to the gym"),
    ("La lune brille intensément", "The moon shines brightly"),
    ("Elle attrape le bus", "She catches the bus"),
    ("Nous saluons nos voisins", "We greet our neighbors"),
    ("Il se peigne les cheveux", "He combs his hair"),
    ("Ils font un signe d'adieu", "They wave goodbye")
]

# Use the entire dataset
pairs = french_to_english

# Split the data for validation (20%)
random.shuffle(pairs)
val_split_idx = int(len(pairs) * 0.2)
val_pairs = pairs[:val_split_idx]
train_pairs = pairs  # Use all pairs for training (100%)

print(f"Training pairs: {len(train_pairs)}")
print(f"Validation pairs: {len(val_pairs)}")

# Improved tokenization function to handle punctuation
def tokenize(text):
    # Replace common punctuation with spaces around them for proper tokenization
    for punct in [".", ",", "!", "?", ";", ":", "'", "'"]:
        text = text.replace(punct, f" {punct} ")
    # Handle special cases for French contractions
    text = text.replace("l'", "l' ")
    text = text.replace("d'", "d' ")
    text = text.replace("j'", "j' ")
    text = text.replace("s'", "s' ")
    text = text.replace("n'", "n' ")
    text = text.replace("qu'", "qu' ")
    # Split by whitespace and filter out empty strings
    tokens = [token for token in text.lower().split() if token]
    return tokens

# Create vocabulary
def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sentence in sentences:
        tokens = tokenize(sentence)
        counter.update(tokens)
    
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

# Generate French and English vocabularies
fr_sentences = [pair[0] for pair in pairs]
eng_sentences = [pair[1] for pair in pairs]

fr_vocab = build_vocab(fr_sentences)
eng_vocab = build_vocab(eng_sentences)

# Create reverse mappings
idx_to_fr = {idx: word for word, idx in fr_vocab.items()}
idx_to_eng = {idx: word for word, idx in eng_vocab.items()}

print(f"French vocabulary size: {len(fr_vocab)}")
print(f"English vocabulary size: {len(eng_vocab)}")

# Updated dataset class with improved tokenization
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
        
        # Convert words to indices using proper tokenization
        src_tokens = tokenize(src_text)
        src_indices = [self.src_vocab.get(token, self.src_vocab['<unk>']) 
                      for token in src_tokens]
        
        tgt_tokens = tokenize(tgt_text)
        tgt_indices = [self.tgt_vocab['<sos>']] + \
                     [self.tgt_vocab.get(token, self.tgt_vocab['<unk>']) 
                      for token in tgt_tokens] + \
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
train_dataset = TranslationDataset(train_pairs, fr_vocab, eng_vocab)
val_dataset = TranslationDataset(val_pairs, fr_vocab, eng_vocab)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

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

# Translation function with improved detokenization
def translate_sentence(model, sentence, src_vocab, tgt_vocab, idx_to_tgt, device, max_len=20):
    model.eval()
    
    # Tokenize and convert to indices
    tokens = tokenize(sentence)
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
    translated_tokens = [idx_to_tgt[idx] for idx in tgt_indices[1:-1] if idx in idx_to_tgt]  # Skip <sos> and <eos>
    
    # Detokenize the output
    translated_text = " ".join(translated_tokens)
    # Fix spaces before punctuation
    for punct in [".", ",", "!", "?", ";", ":"]:
        translated_text = translated_text.replace(f" {punct}", punct)
    # Fix apostrophes
    translated_text = translated_text.replace("' ", "'")
    
    return translated_text

# Function to run grid search on transformer configurations
def run_grid_search():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = []
    
    # Configurations to test
    layers_options = [1, 2, 4]
    heads_options = [2, 4]
    
    for n_layers in layers_options:
        for n_heads in heads_options:
            print(f"\nTraining model with {n_layers} layers and {n_heads} heads")
            
            # Initialize model
            model = TransformerModel(
                src_vocab_size=len(fr_vocab),
                tgt_vocab_size=len(eng_vocab),
                n_heads=n_heads,
                num_layers=n_layers
            )
            
            # Define optimizer and criterion
            optimizer = optim.Adam(model.parameters(), lr=0.0005)
            criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
            
            # Train the model
            train_losses = train_model(model, train_dataloader, optimizer, criterion, n_epochs=50, device=device)
            
            # Evaluate the model
            val_loss, val_accuracy = evaluate_model(model, val_dataloader, criterion, device)
            
            # Evaluate on the entire dataset as well (100% evaluation)
            full_dataloader = DataLoader(TranslationDataset(pairs, fr_vocab, eng_vocab), batch_size=16, shuffle=False)
            full_loss, full_accuracy = evaluate_model(model, full_dataloader, criterion, device)
            
            # Save results
            results.append({
                'n_layers': n_layers,
                'n_heads': n_heads,
                'train_loss': train_losses[-1],
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'full_loss': full_loss,
                'full_accuracy': full_accuracy
            })
            
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            print(f"Full Dataset Loss: {full_loss:.4f}, Accuracy: {full_accuracy:.4f}")
            
            # Test some translations
            test_sentences = [
                "J'ai froid",
                "Elle parle français couramment",
                "Ils visitent la tour Eiffel"
            ]
            
            print("\nSample Translations:")
            for sent in test_sentences:
                translated = translate_sentence(model, sent, fr_vocab, eng_vocab, idx_to_eng, device)
                print(f"French: {sent}")
                print(f"English (predicted): {translated}")
                
                # Find the ground truth if available
                ground_truth = None
                for pair in pairs:
                    if pair[0] == sent:
                        ground_truth = pair[1]
                        break
                
                if ground_truth:
                    print(f"English (actual): {ground_truth}")
                print()
    
    # Plot results for both validation and full dataset
    plt.figure(figsize=(15, 8))
    
    # Extract and organize results for visualization
    labels = [f"{r['n_layers']}L-{r['n_heads']}H" for r in results]
    val_accuracies = [r['val_accuracy'] * 100 for r in results]
    full_accuracies = [r['full_accuracy'] * 100 for r in results]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, val_accuracies, width, label='Validation Set (20%)')
    plt.bar(x + width/2, full_accuracies, width, label='Full Dataset (100%)')
    
    plt.xlabel('Model Configuration')
    plt.ylabel('Accuracy (%)')
    plt.title('Transformer Model Performance Comparison')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('fr2en_model_comparison.png')
    plt.close()
    
    # Create comparison table
    print("\nResults summary:")
    print("-" * 100)
    print(f"{'Model':<20} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<10} {'Full Loss':<12} {'Full Acc':<10}")
    print("-" * 100)
    
    # Sort by full dataset accuracy
    all_results = sorted(results, key=lambda x: x['full_accuracy'], reverse=True)
    
    for r in all_results:
        model_name = f"Transformer {r['n_layers']}L-{r['n_heads']}H"
        print(f"{model_name:<20} {r['train_loss']:<12.4f} {r['val_loss']:<12.4f} {r['val_accuracy']:<10.4f} {r['full_loss']:<12.4f} {r['full_accuracy']:<10.4f}")
    
    return results

if __name__ == "__main__":
    # Run grid search to find the best model configuration
    results = run_grid_search()