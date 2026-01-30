import Database from 'better-sqlite3';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import * as schema from '../src/lib/server/schema';

const sqlite = new Database('data/quest-log.db');

const db = drizzle(sqlite, { schema });

interface TaskData {
	title: string;
	description: string;
	details: string;
}

interface ModuleData {
	name: string;
	description: string;
	tasks: TaskData[];
}

interface PathData {
	name: string;
	description: string;
	language: string;
	color: string;
	skills: string;
	startHint: string;
	difficulty: string;
	estimatedWeeks: number;
	schedule: string;
	modules: ModuleData[];
}

const deepLearningPart2Path: PathData = {
	name: 'Advanced Deep Learning: Entertainment AI & Production',
	description: 'Advanced deep learning topics focusing on entertainment applications, multi-modal models, model interpretability, and production deployment. Part 2 of comprehensive DL guide.',
	language: 'Python',
	color: 'indigo',
	skills: 'multi-modal AI, entertainment models, dialogue generation, model interpretability, knowledge distillation, quantization, production deployment',
	startHint: 'Start with dialogue generation for entertainment applications',
	difficulty: 'advanced',
	estimatedWeeks: 10,
	schedule: `## 10-Week Advanced Deep Learning Schedule

### Weeks 1-3: Entertainment AI Models

#### Week 1: Dialogue & Scene Generation
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Setup | Review transformer architectures, prepare datasets |
| Tue | Dialogue Model | Implement MovieDialogueGenerator with conditioning |
| Wed | Scene Generation | Build multi-output scene generator |
| Thu | Training | Train dialogue model on movie scripts |
| Fri | Evaluation | Test generation quality, coherence |
| Weekend | Refinement | Improve conditioning mechanisms |

#### Week 2: Multi-Modal Models
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Visual Encoding | Build visual encoder for video frames |
| Wed-Thu | Audio Encoding | Implement audio feature extraction |
| Fri | Text Encoding | Set up text/subtitle encoder |
| Weekend | Fusion | Build multi-modal fusion architecture |

#### Week 3: Complete Movie Understanding
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Pipeline | Implement full MovieUnderstandingPipeline |
| Wed-Thu | Temporal Analysis | Add scene segmentation, emotion detection |
| Fri | Integration | Connect all components |
| Weekend | Testing | Process sample videos, evaluate outputs |

### Weeks 4-5: Training Advanced Techniques

#### Week 4: Curriculum & Self-Paced Learning
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Curriculum Learning | Implement progressive difficulty training |
| Wed-Thu | Self-Paced Learning | Add dynamic sample weighting |
| Fri | Comparison | Compare standard vs curriculum training |
| Weekend | Analysis | Analyze learning curves, convergence |

#### Week 5: Mixed Precision & Optimization
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Mixed Precision | Implement FP16 training with loss scaling |
| Wed-Thu | Gradient Accumulation | Handle large effective batch sizes |
| Fri | Benchmarking | Compare training speeds, memory usage |
| Weekend | Optimization | Fine-tune hyperparameters |

### Weeks 6-7: Model Interpretability

#### Week 6: Attention Visualization & Probing
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Attention Viz | Visualize attention patterns |
| Wed-Thu | Probing Classifiers | Train probes on representations |
| Fri | Analysis | Understand what layers encode |
| Weekend | Documentation | Document findings |

#### Week 7: Advanced Interpretability
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | CAV Implementation | Build Concept Activation Vectors |
| Wed-Thu | Concept Analysis | Test concept sensitivity |
| Fri | Integration | Combine interpretability methods |
| Weekend | Case Studies | Analyze specific model behaviors |

### Weeks 8-10: Production Deployment

#### Week 8: Model Compression
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Knowledge Distillation | Train student from teacher model |
| Wed-Thu | Pruning | Implement magnitude and structured pruning |
| Fri | Quantization | Apply dynamic and static quantization |
| Weekend | Comparison | Benchmark size, speed, accuracy trade-offs |

#### Week 9: Deployment Optimizations
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | TorchScript | Export models to TorchScript |
| Tue | ONNX | Convert to ONNX format |
| Wed | Torch Compile | Apply PyTorch 2.0 optimizations |
| Thu-Fri | API Server | Build FastAPI inference server |
| Weekend | Load Testing | Benchmark throughput, latency |

#### Week 10: Production Pipeline
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Containerization | Dockerize inference service |
| Wed-Thu | Monitoring | Add logging, metrics, alerts |
| Fri | CI/CD | Set up deployment pipeline |
| Weekend | Documentation | Complete production deployment guide |

### Daily Commitment: 2-3 hours
### Prerequisites: Understanding of basic transformers, PyTorch`,
	modules: [
		{
			name: 'Entertainment AI: Generative Models',
			description: 'Build AI models for entertainment applications including dialogue and scene generation',
			tasks: [
				{
					title: 'Implement MovieDialogueGenerator with character conditioning',
					description: 'Create dialogue generation model conditioned on character traits and emotions',
					details: `## Movie Dialogue Generation

### Architecture Overview

Generate contextually appropriate dialogue conditioned on:
- Character personality/traits
- Scene context
- Emotional state
- Previous dialogue history

### Implementation

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MovieDialogueGenerator(nn.Module):
    """
    Generate dialogue for characters in movies/TV shows.

    Conditioning on:
    - Character personality/traits
    - Scene context
    - Emotional state
    - Previous dialogue
    """
    def __init__(self, vocab_size, d_model=768, num_characters=100,
                 num_emotions=10, num_layers=8, num_heads=12):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)
        self.character_emb = nn.Embedding(num_characters, d_model)
        self.emotion_emb = nn.Embedding(num_emotions, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, character_id, emotion_id, context_emb=None):
        """
        Args:
            input_ids: (batch, seq_len) - token indices
            character_id: (batch,) - character ID
            emotion_id: (batch,) - emotion ID
            context_emb: (batch, context_len, d_model) - optional scene context

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.token_emb(input_ids)  # (B, T, d_model)

        # Positional embeddings
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(positions)

        # Add character conditioning (broadcast across sequence)
        char_emb = self.character_emb(character_id).unsqueeze(1)  # (B, 1, d_model)
        x = x + char_emb

        # Add emotion conditioning
        emot_emb = self.emotion_emb(emotion_id).unsqueeze(1)  # (B, 1, d_model)
        x = x + emot_emb

        x = self.dropout(x)

        # Scene context as memory (if provided)
        if context_emb is None:
            context_emb = torch.zeros(B, 1, self.d_model, device=device)

        # Causal mask (can only attend to previous tokens)
        causal_mask = torch.triu(
            torch.ones(T, T, device=device),
            diagonal=1
        ).bool()

        # Transformer decoder
        x = self.decoder(x, context_emb, tgt_mask=causal_mask)

        # Project to vocabulary
        logits = self.output(x)

        return logits

    def generate(self, start_tokens, character_id, emotion_id, context_emb=None,
                 max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        """
        Generate dialogue using sampling strategies.

        Args:
            start_tokens: (batch, start_len) - initial tokens
            character_id: (batch,) - character ID
            emotion_id: (batch,) - emotion ID
            context_emb: optional scene context
            max_length: maximum generation length
            temperature: sampling temperature (higher = more random)
            top_k: keep only top k tokens
            top_p: nucleus sampling threshold

        Returns:
            generated: (batch, max_length) - generated token IDs
        """
        self.eval()

        batch_size = start_tokens.size(0)
        generated = start_tokens

        with torch.no_grad():
            for _ in range(max_length - start_tokens.size(1)):
                # Get logits for current sequence
                logits = self.forward(generated, character_id, emotion_id, context_emb)

                # Get logits for next token
                next_token_logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Check for end-of-sequence token (optional)
                # if (next_token == eos_token_id).all():
                #     break

        return generated


# Training example
class DialogueTrainer:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-5,
            weight_decay=0.01
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000
        )

    def train_step(self, batch):
        """Single training step."""
        input_ids = batch['input_ids'].to(self.device)
        character_ids = batch['character_id'].to(self.device)
        emotion_ids = batch['emotion_id'].to(self.device)
        context_emb = batch.get('context_emb', None)

        if context_emb is not None:
            context_emb = context_emb.to(self.device)

        # Forward pass
        logits = self.model(input_ids[:, :-1], character_ids, emotion_ids, context_emb)

        # Compute loss (predict next token)
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                character_ids = batch['character_id'].to(self.device)
                emotion_ids = batch['emotion_id'].to(self.device)
                context_emb = batch.get('context_emb', None)

                if context_emb is not None:
                    context_emb = context_emb.to(self.device)

                logits = self.model(input_ids[:, :-1], character_ids, emotion_ids, context_emb)
                targets = input_ids[:, 1:]

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=self.tokenizer.pad_token_id,
                    reduction='sum'
                )

                total_loss += loss.item()

        self.model.train()
        return total_loss / len(val_loader.dataset)


# Usage example
if __name__ == "__main__":
    # Initialize model
    model = MovieDialogueGenerator(
        vocab_size=50000,
        d_model=768,
        num_characters=100,
        num_emotions=10
    )

    # Example generation
    start_tokens = torch.tensor([[1, 2, 3]])  # "Hello, how"
    character_id = torch.tensor([5])  # Character 5 (e.g., "Tony Stark")
    emotion_id = torch.tensor([2])    # Emotion 2 (e.g., "confident")

    generated = model.generate(
        start_tokens,
        character_id,
        emotion_id,
        max_length=50,
        temperature=0.8,
        top_p=0.9
    )

    print("Generated dialogue:", generated)
\`\`\`

### Character & Emotion Embeddings

**Character embedding captures:**
- Speaking style
- Vocabulary preferences
- Personality traits
- Typical sentence structure

**Emotion embedding captures:**
- Emotional tone
- Word choice variations
- Intensity of expression

### Training Data Format

\`\`\`python
# Dataset structure
dialogue_data = {
    'input_ids': [101, 2023, 2003, 1037, 2844, ...],  # Tokenized dialogue
    'character_id': 5,  # Character identifier
    'emotion_id': 2,    # Emotion identifier
    'context_emb': [...],  # Optional: scene description embedding
}
\`\`\`

### Advanced Features

**1. Multi-turn dialogue:**
\`\`\`python
# Include conversation history in context
context = encode_previous_turns(previous_dialogue)
logits = model(current_input, character_id, emotion_id, context)
\`\`\`

**2. Style transfer:**
\`\`\`python
# Generate same content in different character's style
original_character = 1
target_character = 5
logits = model(input_ids, target_character, emotion_id)
\`\`\`

**3. Emotion control:**
\`\`\`python
# Generate dialogue transitioning between emotions
for emotion_id in range(angry, calm):
    generated = model.generate(start, char_id, emotion_id)
\`\`\`

### Evaluation Metrics

- **Perplexity:** How well model predicts next token
- **BLEU score:** Compare to reference dialogue
- **Character consistency:** Does it match character's style?
- **Emotional appropriateness:** Does emotion match context?
- **Human evaluation:** Quality, coherence, entertainment value

### Practice Exercises

- [ ] Implement character embedding visualization
- [ ] Train on movie script dataset
- [ ] Generate multi-turn conversations
- [ ] Implement emotion interpolation
- [ ] Build interactive dialogue demo`
				},
				{
					title: 'Build multi-modal fusion for movie understanding',
					description: 'Combine visual, audio, and text modalities for comprehensive video analysis',
					details: `## Multi-Modal Movie Understanding

### Architecture Overview

Fuse three modalities to understand movies:
1. **Visual:** Frames, scenes, objects, faces
2. **Audio:** Speech, music, sound effects
3. **Text:** Subtitles, dialogue, narration

### Complete Implementation

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchaudio

class VisualEncoder(nn.Module):
    """Encode video frames using pretrained CNN + temporal model."""

    def __init__(self, feature_dim=768):
        super().__init__()

        # Pretrained ResNet for frame features
        resnet = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze early layers
        for param in list(self.cnn.parameters())[:-10]:
            param.requires_grad = False

        # Temporal modeling with LSTM
        self.temporal = nn.LSTM(
            input_size=2048,
            hidden_size=feature_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.feature_dim = feature_dim

    def forward(self, frames):
        """
        Args:
            frames: (batch, num_frames, C, H, W)

        Returns:
            features: (batch, num_frames, feature_dim)
        """
        B, T, C, H, W = frames.shape

        # Extract frame features
        frames_flat = frames.view(B * T, C, H, W)
        frame_features = self.cnn(frames_flat)  # (B*T, 2048, 1, 1)
        frame_features = frame_features.view(B, T, -1)  # (B, T, 2048)

        # Temporal modeling
        temporal_features, _ = self.temporal(frame_features)

        return temporal_features


class AudioEncoder(nn.Module):
    """Encode audio features."""

    def __init__(self, feature_dim=256):
        super().__init__()

        # Mel spectrogram
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80
        )

        # CNN for spectrogram
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))
        )

        # Temporal model
        self.temporal = nn.LSTM(
            128,
            feature_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, audio):
        """
        Args:
            audio: (batch, num_samples) - raw audio waveform

        Returns:
            features: (batch, time_steps, feature_dim)
        """
        # Convert to mel spectrogram
        mel = self.mel_spec(audio)  # (B, n_mels, time)
        mel = mel.unsqueeze(1)  # (B, 1, n_mels, time)

        # CNN features
        conv_out = self.conv(mel)  # (B, 128, 1, time)
        conv_out = conv_out.squeeze(2).transpose(1, 2)  # (B, time, 128)

        # Temporal modeling
        features, _ = self.temporal(conv_out)

        return features


class TextEncoder(nn.Module):
    """Encode text/subtitles using transformer."""

    def __init__(self, vocab_size=50000, feature_dim=768):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, feature_dim)
        self.pos_encoding = nn.Embedding(512, feature_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=feature_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len)

        Returns:
            features: (batch, seq_len, feature_dim)
        """
        B, T = input_ids.shape

        # Embeddings
        x = self.embedding(input_ids)
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_encoding(positions)

        # Transformer
        features = self.transformer(x)

        return features


class MultiModalFusion(nn.Module):
    """
    Fuse visual, audio, and text modalities using cross-modal attention.
    """
    def __init__(self, visual_dim=768, audio_dim=256, text_dim=768, fusion_dim=512):
        super().__init__()

        # Project to common dimension
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        # Cross-modal attention (each modality attends to others)
        self.v_to_a = nn.MultiheadAttention(fusion_dim, 8, batch_first=True)
        self.v_to_t = nn.MultiheadAttention(fusion_dim, 8, batch_first=True)
        self.a_to_v = nn.MultiheadAttention(fusion_dim, 8, batch_first=True)
        self.a_to_t = nn.MultiheadAttention(fusion_dim, 8, batch_first=True)
        self.t_to_v = nn.MultiheadAttention(fusion_dim, 8, batch_first=True)
        self.t_to_a = nn.MultiheadAttention(fusion_dim, 8, batch_first=True)

        # Layer norms
        self.norm_v = nn.LayerNorm(fusion_dim)
        self.norm_a = nn.LayerNorm(fusion_dim)
        self.norm_t = nn.LayerNorm(fusion_dim)

        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

    def forward(self, visual, audio, text=None):
        """
        Args:
            visual: (batch, time, visual_dim)
            audio: (batch, time, audio_dim)
            text: (batch, seq_len, text_dim) - optional

        Returns:
            fused: (batch, time, fusion_dim)
        """
        # Project to common dimension
        v = self.visual_proj(visual)
        a = self.audio_proj(audio)
        t = self.text_proj(text) if text is not None else torch.zeros_like(v)

        # Cross-modal attention
        # Visual attends to audio and text
        v_from_a, _ = self.v_to_a(v, a, a)
        v_from_t, _ = self.v_to_t(v, t, t)
        v_attended = self.norm_v(v + v_from_a + v_from_t)

        # Audio attends to visual and text
        a_from_v, _ = self.a_to_v(a, v, v)
        a_from_t, _ = self.a_to_t(a, t, t)
        a_attended = self.norm_a(a + a_from_v + a_from_t)

        # Text attends to visual and audio
        t_from_v, _ = self.t_to_v(t, v, v)
        t_from_a, _ = self.t_to_a(t, a, a)
        t_attended = self.norm_t(t + t_from_v + t_from_a)

        # Concatenate and fuse
        combined = torch.cat([v_attended, a_attended, t_attended], dim=-1)
        fused = self.fusion_mlp(combined)

        return fused


class MovieUnderstandingPipeline:
    """
    Complete pipeline for understanding movies.

    Components:
    1. Visual: Scene classification, object detection, face recognition
    2. Audio: Speech recognition, music analysis, sound effects
    3. Text: Subtitle analysis, plot understanding
    4. Temporal: Scene segmentation, narrative structure
    """
    def __init__(self, device='cuda'):
        self.device = device

        # Encoders
        self.visual_encoder = VisualEncoder().to(device)
        self.audio_encoder = AudioEncoder().to(device)
        self.text_encoder = TextEncoder().to(device)

        # Fusion
        self.fusion_model = MultiModalFusion().to(device)

        # Downstream tasks
        self.scene_classifier = nn.Linear(512, 20).to(device)  # 20 scene types
        self.emotion_detector = nn.Linear(512, 8).to(device)   # 8 emotions

    def process_video(self, video_path, subtitles_path=None):
        """
        Process entire video.

        Returns:
        - Scene-level features
        - Detected emotions
        - Scene boundaries
        - Key moments
        """
        import cv2

        # Extract frames (simplified - in practice use ffmpeg)
        frames = self.extract_frames(video_path, fps=1)

        # Extract audio
        audio = self.extract_audio(video_path)

        # Load subtitles
        subtitles = self.load_subtitles(subtitles_path) if subtitles_path else None

        # Encode each modality
        with torch.no_grad():
            visual_features = self.visual_encoder(frames)
            audio_features = self.audio_encoder(audio)
            text_features = self.text_encoder(subtitles) if subtitles else None

            # Fuse modalities
            fused_features = self.fusion_model(visual_features, audio_features, text_features)

            # Scene classification
            scene_probs = F.softmax(self.scene_classifier(fused_features), dim=-1)

            # Emotion detection
            emotion_probs = F.softmax(self.emotion_detector(fused_features), dim=-1)

        return {
            'fused_features': fused_features,
            'scene_predictions': scene_probs,
            'emotion_predictions': emotion_probs
        }

    def extract_frames(self, video_path, fps=1):
        """Extract frames from video at specified FPS."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)

        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Resize and normalize
                frame = cv2.resize(frame, (224, 224))
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames.append(frame)

            frame_count += 1

        cap.release()
        return torch.stack(frames).unsqueeze(0).to(self.device)

    def extract_audio(self, video_path, sample_rate=16000):
        """Extract audio from video."""
        import subprocess
        import tempfile

        # Extract audio using ffmpeg
        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            subprocess.run([
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(sample_rate), '-ac', '1',
                f.name
            ], check=True, capture_output=True)

            waveform, sr = torchaudio.load(f.name)

        return waveform.to(self.device)

    def load_subtitles(self, subtitles_path):
        """Load and tokenize subtitles."""
        # Simplified - in practice parse SRT/VTT files
        with open(subtitles_path, 'r') as f:
            text = f.read()

        # Tokenize (placeholder - use actual tokenizer)
        tokens = [ord(c) % 50000 for c in text[:512]]
        return torch.tensor([tokens]).to(self.device)
\`\`\`

### Training Multi-Modal Models

\`\`\`python
# Training loop
def train_multimodal(model, train_loader, epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for batch in train_loader:
            frames = batch['frames'].cuda()
            audio = batch['audio'].cuda()
            subtitles = batch['subtitles'].cuda()
            labels = batch['labels'].cuda()

            # Forward pass
            visual_feat = model.visual_encoder(frames)
            audio_feat = model.audio_encoder(audio)
            text_feat = model.text_encoder(subtitles)

            fused = model.fusion_model(visual_feat, audio_feat, text_feat)

            # Task-specific heads
            scene_pred = model.scene_classifier(fused.mean(dim=1))

            # Loss
            loss = F.cross_entropy(scene_pred, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
\`\`\`

### Applications

1. **Automatic scene detection**
2. **Emotion arc tracking**
3. **Content-based search**
4. **Automatic highlight generation**
5. **Accessibility (audio description generation)**

### Practice Exercises

- [ ] Implement complete pipeline
- [ ] Test on sample movie clips
- [ ] Visualize cross-modal attention
- [ ] Build scene search system
- [ ] Create emotion timeline visualization`
				}
			]
		},
		{
			name: 'Model Interpretability',
			description: 'Understand what your models learn and how they make decisions',
			tasks: [
				{
					title: 'Implement Concept Activation Vectors (CAV) for interpretability',
					description: 'Build TCAV to understand what concepts models use for predictions',
					details: String.raw`## Concept Activation Vectors (TCAV)

### What is TCAV?

Testing with Concept Activation Vectors helps answer: "Does my model use concept X to make prediction Y?"

Example questions:
- Does the model use "striped fur" for tiger classification?
- Does sentiment classifier rely on negation words?
- Does face detector use "eyes" or just "skin tone"?

### Implementation

` + `\`\`\`python
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression

class ConceptActivationVectors:
    """
    TCAV: Testing with Concept Activation Vectors

    Understand what concepts a model uses for predictions.
    """
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.activations = {}

        # Register hook to capture activations
        self._register_hook()

    def _register_hook(self):
        """Register forward hook to capture layer activations."""
        def hook(module, input, output):
            self.activations[self.layer_name] = output.detach()

        # Find the layer and register hook
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook)
                print(f"Hook registered for layer: {name}")
                break

    def get_activations(self, inputs):
        """
        Get activations for inputs at the specified layer.

        Args:
            inputs: batch of inputs

        Returns:
            activations: layer activations
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(inputs)
        return self.activations[self.layer_name]

    def train_concept_classifier(self, concept_examples, random_examples):
        """
        Train linear classifier to separate concept from random examples.

        The CAV is the normal vector to the decision boundary.

        Args:
            concept_examples: images containing the concept
            random_examples: random images without the concept

        Returns:
            cav: concept activation vector (normalized weight vector)
            accuracy: classifier accuracy
        """
        # Get activations
        concept_activations = self.get_activations(concept_examples)
        random_activations = self.get_activations(random_examples)

        # Flatten activations
        concept_flat = concept_activations.view(len(concept_examples), -1).cpu().numpy()
        random_flat = random_activations.view(len(random_examples), -1).cpu().numpy()

        # Prepare training data
        X = np.vstack([concept_flat, random_flat])
        y = np.array([1] * len(concept_examples) + [0] * len(random_examples))

        # Train linear classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)

        accuracy = clf.score(X, y)
        print(f"CAV classifier accuracy: {accuracy:.3f}")

        # CAV is the weight vector (normalized)
        cav = clf.coef_[0]
        cav = cav / np.linalg.norm(cav)

        return cav, accuracy

    def conceptual_sensitivity(self, inputs, cav, class_idx):
        """
        Compute sensitivity of predictions to concept.

        TCAV score = fraction of examples where concept positively
        influences the target class prediction.

        Args:
            inputs: test inputs
            cav: concept activation vector
            class_idx: target class index

        Returns:
            tcav_score: sensitivity to concept (0 to 1)
            sensitivities: per-example sensitivity values
        """
        inputs = inputs.requires_grad_(True)

        # Forward pass
        activations = self.get_activations(inputs)
        outputs = self.model(inputs)

        # Get logits for target class
        target_logits = outputs[:, class_idx]

        # Compute gradients of target class w.r.t. activations
        grads = torch.autograd.grad(
            outputs=target_logits.sum(),
            inputs=activations,
            retain_graph=True
        )[0]

        # Flatten gradients
        grads_flat = grads.view(grads.size(0), -1).cpu().numpy()

        # Compute directional derivative (sensitivity)
        # sensitivity = grad · CAV
        cav_expanded = np.expand_dims(cav, 0)
        sensitivities = np.sum(grads_flat * cav_expanded, axis=1)

        # TCAV score: fraction of positive sensitivities
        tcav_score = np.mean(sensitivities > 0)

        return tcav_score, sensitivities

    def generate_report(self, concept_name, inputs, cav, class_idx, class_name):
        """
        Generate interpretability report.

        Args:
            concept_name: name of the concept
            inputs: test inputs
            cav: concept activation vector
            class_idx: target class index
            class_name: name of target class

        Returns:
            report: dictionary with analysis results
        """
        tcav_score, sensitivities = self.conceptual_sensitivity(inputs, cav, class_idx)

        report = {
            'concept': concept_name,
            'class': class_name,
            'tcav_score': tcav_score,
            'interpretation': self._interpret_score(tcav_score),
            'positive_examples': np.sum(sensitivities > 0),
            'negative_examples': np.sum(sensitivities < 0),
            'mean_sensitivity': np.mean(sensitivities),
            'std_sensitivity': np.std(sensitivities)
        }

        return report

    def _interpret_score(self, tcav_score):
        """Interpret TCAV score."""
        if tcav_score > 0.8:
            return "Strong positive influence"
        elif tcav_score > 0.6:
            return "Moderate positive influence"
        elif tcav_score > 0.4:
            return "Weak influence"
        elif tcav_score > 0.2:
            return "Moderate negative influence"
        else:
            return "Strong negative influence"


# Example usage
def example_tcav_analysis():
    """
    Example: Does a tiger classifier use "striped fur" concept?
    """
    import torchvision.models as models

    # Load pretrained model
    model = models.resnet50(pretrained=True)
    model.eval()

    # Initialize TCAV
    tcav = ConceptActivationVectors(model, layer_name='layer4')

    # Prepare concept examples
    # concept_images: images with striped patterns
    # random_images: random images without stripes
    concept_images = torch.randn(50, 3, 224, 224)  # Placeholder
    random_images = torch.randn(50, 3, 224, 224)   # Placeholder

    # Train CAV
    cav, accuracy = tcav.train_concept_classifier(concept_images, random_images)

    if accuracy < 0.7:
        print("Warning: Low CAV accuracy. Concept may not be well-defined.")

    # Test on tiger images
    tiger_images = torch.randn(100, 3, 224, 224)  # Placeholder
    tiger_class_idx = 292  # ImageNet tiger class

    # Compute TCAV score
    report = tcav.generate_report(
        'striped_fur',
        tiger_images,
        cav,
        tiger_class_idx,
        'tiger'
    )

    print("\n=== TCAV Analysis Report ===")
    print(f"Concept: {report['concept']}")
    print(f"Class: {report['class']}")
    print(f"TCAV Score: {report['tcav_score']:.3f}")
    print(f"Interpretation: {report['interpretation']}")
    print(f"Positive examples: {report['positive_examples']}")
    print(f"Negative examples: {report['negative_examples']}")

    return report


# Multiple concepts comparison
def compare_concepts(model, layer_name, concepts, test_images, class_idx, class_name):
    """
    Compare influence of multiple concepts.

    Args:
        model: neural network
        layer_name: which layer to analyze
        concepts: dict of {name: (concept_imgs, random_imgs)}
        test_images: images to test on
        class_idx: target class
        class_name: class name
    """
    tcav = ConceptActivationVectors(model, layer_name)

    results = {}

    for concept_name, (concept_imgs, random_imgs) in concepts.items():
        print(f"\nAnalyzing concept: {concept_name}")

        # Train CAV
        cav, acc = tcav.train_concept_classifier(concept_imgs, random_imgs)

        # Generate report
        report = tcav.generate_report(
            concept_name,
            test_images,
            cav,
            class_idx,
            class_name
        )

        results[concept_name] = report

    # Print comparison
    print("\n=== Concept Comparison ===")
    for name, report in sorted(results.items(), key=lambda x: x[1]['tcav_score'], reverse=True):
        print(f"{name:20s}: {report['tcav_score']:.3f} ({report['interpretation']})")

    return results


if __name__ == "__main__":
    example_tcav_analysis()
\`\`\`` + `

### Understanding TCAV Scores

**TCAV Score Interpretation:**
- **> 0.8:** Concept strongly influences predictions
- **0.5-0.8:** Moderate positive influence
- **0.4-0.6:** Weak or no influence
- **0.2-0.4:** Moderate negative influence
- **< 0.2:** Concept strongly suppresses predictions

### Applications

1. **Debugging:** Find unwanted biases
2. **Validation:** Ensure model uses correct features
3. **Trust:** Explain decisions to stakeholders
4. **Improvement:** Identify missing concepts

### Best Practices

**Concept Selection:**
- Use well-defined, specific concepts
- Ensure concept images are diverse
- Use enough examples (50+ recommended)

**Random Examples:**
- Should be diverse
- Should not contain the concept
- Should be from similar distribution

**Statistical Testing:**
- Run multiple random trials
- Compute confidence intervals
- Test significance of TCAV scores

### Practice Exercises

- [ ] Implement TCAV for image classifier
- [ ] Test multiple concepts on same class
- [ ] Visualize concept direction in activation space
- [ ] Compare TCAV across different layers
- [ ] Identify and remove biased concepts`
				}
			]
		},
		{
			name: 'Production Deployment',
			description: 'Deploy models to production with optimization and monitoring',
			tasks: [
				{
					title: 'Implement knowledge distillation for model compression',
					description: 'Train smaller student model to mimic larger teacher model',
					details: `## Knowledge Distillation

### Concept

Train a smaller "student" model to mimic a larger "teacher" model by learning from the teacher's soft predictions.

**Benefits:**
- Smaller model size (10-100x reduction)
- Faster inference
- Similar accuracy to teacher
- Transfers "dark knowledge" from teacher

### Implementation

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillation:
    """
    Train smaller student model to mimic larger teacher.

    Loss = α × KL(soft_student || soft_teacher) + (1-α) × CE(student, labels)

    Soft labels (high temperature) transfer "dark knowledge"
    """
    def __init__(self, teacher, student, temperature=4.0, alpha=0.5):
        """
        Args:
            teacher: large pretrained model
            student: smaller model to train
            temperature: softmax temperature for soft labels
            alpha: weight for distillation loss (0=hard labels only, 1=soft labels only)
        """
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Compute combined distillation + hard label loss.

        Args:
            student_logits: raw student outputs (batch, num_classes)
            teacher_logits: raw teacher outputs (batch, num_classes)
            labels: true labels (batch,)

        Returns:
            loss: combined loss
        """
        # Soft labels from teacher (high temperature)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Soft predictions from student
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)

        # KL divergence for distillation
        # Scale by T^2 to account for temperature
        distill_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard label loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_logits, labels)

        # Combined loss
        loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss

        return loss, distill_loss.item(), hard_loss.item()

    def train_epoch(self, train_loader, optimizer, device='cuda'):
        """Train student for one epoch."""
        self.student.train()
        self.teacher.eval()

        total_loss = 0
        total_distill = 0
        total_hard = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Teacher predictions (no gradients)
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)

            # Student predictions
            student_logits = self.student(inputs)

            # Compute loss
            loss, distill, hard = self.distillation_loss(
                student_logits,
                teacher_logits,
                labels
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_distill += distill
            total_hard += hard

        num_batches = len(train_loader)
        return {
            'total_loss': total_loss / num_batches,
            'distill_loss': total_distill / num_batches,
            'hard_loss': total_hard / num_batches
        }

    def train(self, train_loader, val_loader, epochs=10, lr=1e-4, device='cuda'):
        """Full training loop."""
        self.student = self.student.to(device)
        self.teacher = self.teacher.to(device)

        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, device)

            # Validate
            val_acc = self.evaluate(val_loader, device)

            # Learning rate schedule
            scheduler.step()

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - Total: {train_metrics['total_loss']:.4f}, "
                  f"Distill: {train_metrics['distill_loss']:.4f}, "
                  f"Hard: {train_metrics['hard_loss']:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.student.state_dict(), 'best_student.pt')
                print(f"  New best model saved!")

        print(f"\nBest validation accuracy: {best_val_acc:.4f}")
        return self.student

    def evaluate(self, val_loader, device='cuda'):
        """Evaluate student on validation set."""
        self.student.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.student(inputs)
                predictions = outputs.argmax(dim=-1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        self.student.train()
        return accuracy


# Example: Distill ResNet50 to ResNet18
def example_distillation():
    import torchvision.models as models

    # Teacher: Large model
    teacher = models.resnet50(pretrained=True)

    # Student: Smaller model
    student = models.resnet18(pretrained=False)

    # Distillation trainer
    distiller = KnowledgeDistillation(
        teacher=teacher,
        student=student,
        temperature=4.0,
        alpha=0.7  # Weight distillation more than hard labels
    )

    # Train (assuming train_loader and val_loader exist)
    # trained_student = distiller.train(train_loader, val_loader, epochs=20)

    # Compare sizes
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Compression ratio: {teacher_params / student_params:.1f}x")


# Advanced: Feature-based distillation
class FeatureDistillation(KnowledgeDistillation):
    """
    Distillation using intermediate features, not just final outputs.

    Student learns to match teacher's internal representations.
    """
    def __init__(self, teacher, student, temperature=4.0, alpha=0.5,
                 teacher_layers=None, student_layers=None, beta=0.5):
        super().__init__(teacher, student, temperature, alpha)

        self.teacher_layers = teacher_layers or []
        self.student_layers = student_layers or []
        self.beta = beta  # Weight for feature matching loss

        # Hooks to capture intermediate features
        self.teacher_features = {}
        self.student_features = {}

        self._register_feature_hooks()

    def _register_feature_hooks(self):
        """Register hooks to capture intermediate layer outputs."""
        def get_teacher_hook(name):
            def hook(module, input, output):
                self.teacher_features[name] = output.detach()
            return hook

        def get_student_hook(name):
            def hook(module, input, output):
                self.student_features[name] = output
            return hook

        # Register teacher hooks
        for name, module in self.teacher.named_modules():
            if name in self.teacher_layers:
                module.register_forward_hook(get_teacher_hook(name))

        # Register student hooks
        for name, module in self.student.named_modules():
            if name in self.student_layers:
                module.register_forward_hook(get_student_hook(name))

    def feature_matching_loss(self):
        """Compute loss for matching intermediate features."""
        total_loss = 0

        for t_layer, s_layer in zip(self.teacher_layers, self.student_layers):
            t_feat = self.teacher_features[t_layer]
            s_feat = self.student_features[s_layer]

            # Adaptive pooling if shapes don't match
            if t_feat.shape != s_feat.shape:
                s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:])

            # MSE loss
            loss = F.mse_loss(s_feat, t_feat)
            total_loss += loss

        return total_loss / len(self.teacher_layers)

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Combined output + feature distillation."""
        # Output distillation
        output_loss, distill, hard = super().distillation_loss(
            student_logits, teacher_logits, labels
        )

        # Feature matching
        feature_loss = self.feature_matching_loss()

        # Combine
        total_loss = output_loss + self.beta * feature_loss

        return total_loss, distill, hard
\`\`\`

### Why Temperature Matters

**Low temperature (T=1):**
- Hard labels: argmax probability ≈ 1, others ≈ 0
- Less information transfer

**High temperature (T=4-10):**
- Soft labels: probabilities more evenly distributed
- Transfers "dark knowledge" (relative similarities)
- Student learns from teacher's uncertainties

**Example:**
\`\`\`python
# Hard labels (T=1)
[0.95, 0.03, 0.02]  # "Dog" class dominates

# Soft labels (T=4)
[0.70, 0.20, 0.10]  # "Dog" still highest, but cat/wolf have info
\`\`\`

### Compression Results

Typical results on ImageNet:

| Teacher | Student | Accuracy Drop | Speedup |
|---------|---------|---------------|---------|
| ResNet-152 | ResNet-50 | 1-2% | 3x |
| ResNet-50 | ResNet-18 | 2-3% | 2x |
| BERT-Large | BERT-Base | 1-2% | 2x |
| GPT-3 | GPT-2 | Variable | 100x+ |

### Practice Exercises

- [ ] Implement basic distillation
- [ ] Experiment with different temperatures
- [ ] Try feature-based distillation
- [ ] Distill transformer model
- [ ] Measure size/speed/accuracy tradeoffs
- [ ] Deploy distilled model to mobile device`
				},
				{
					title: 'Deploy model with FastAPI and optimize for production',
					description: 'Build production-ready inference API with monitoring and optimization',
					details: `## Production Model Deployment

### Complete Inference Server with FastAPI

\`\`\`python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import torch.nn as nn
import uvicorn
import time
import logging
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total number of requests')
REQUEST_LATENCY = Histogram('model_request_latency_seconds', 'Request latency')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions', ['class'])

# FastAPI app
app = FastAPI(
    title="ML Model API",
    description="Production model inference API",
    version="1.0.0"
)

# Global model storage
model = None
device = None

# Request/Response models
class PredictionRequest(BaseModel):
    data: List[List[float]] = Field(..., description="Input data as nested list")
    return_probabilities: bool = Field(False, description="Return class probabilities")

    class Config:
        schema_extra = {
            "example": {
                "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "return_probabilities": True
            }
        }

class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    latency_ms: float
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str

# Startup: Load model
@app.on_event("startup")
async def load_model():
    """Load model at startup."""
    global model, device

    logger.info("Loading model...")

    try:
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Load model (TorchScript for production)
        model_path = 'model.pt'
        model = torch.jit.load(model_path, map_location=device)
        model.eval()

        # Warm up
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model(dummy_input)

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device),
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions on input data.
    """
    # Metrics
    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Prepare input
        inputs = torch.tensor(request.data, dtype=torch.float32)

        if inputs.dim() == 2:
            # Assume image data, reshape appropriately
            # This is simplified - in practice, handle different input shapes
            inputs = inputs.view(-1, 3, 224, 224)

        inputs = inputs.to(device)

        # Inference
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=-1)
            predictions = probs.argmax(dim=-1).cpu().tolist()

        # Track predictions
        for pred in predictions:
            PREDICTION_COUNT.labels(class=str(pred)).inc()

        # Prepare response
        response = PredictionResponse(
            predictions=predictions,
            probabilities=probs.cpu().tolist() if request.return_probabilities else None,
            latency_ms=(time.time() - start_time) * 1000,
            model_version="1.0.0"
        )

        # Record latency
        REQUEST_LATENCY.observe(time.time() - start_time)

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "ML Model API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
\`\`\`

### Dockerfile for Deployment

\`\`\`dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY model.pt .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
\`\`\`

### Docker Compose with Monitoring

\`\`\`yaml
version: '3.8'

services:
  model-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model.pt
    volumes:
      - ./model.pt:/app/model.pt:ro
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
\`\`\`

### Prometheus Configuration

\`\`\`yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'model-api'
    static_configs:
      - targets: ['model-api:8000']
\`\`\`

### Model Export to TorchScript

\`\`\`python
import torch

def export_to_torchscript(model, example_input, save_path):
    """
    Export PyTorch model to TorchScript for production.

    TorchScript benefits:
    - No Python dependency
    - Optimized for inference
    - Can run in C++
    - Better performance
    """
    model.eval()

    # Trace model (for models without control flow)
    traced = torch.jit.trace(model, example_input)

    # Or script (for models with if/loops)
    # scripted = torch.jit.script(model)

    # Optimize for inference
    traced = torch.jit.optimize_for_inference(traced)

    # Freeze (inline parameters)
    traced = torch.jit.freeze(traced)

    # Save
    traced.save(save_path)

    print(f"Model exported to {save_path}")

    # Verify
    loaded = torch.jit.load(save_path)
    with torch.no_grad():
        original_out = model(example_input)
        loaded_out = loaded(example_input)
        print(f"Max difference: {(original_out - loaded_out).abs().max().item()}")

    return traced

# Export model
model = YourModel()
model.load_state_dict(torch.load('checkpoint.pt'))
example_input = torch.randn(1, 3, 224, 224)

export_to_torchscript(model, example_input, 'model.pt')
\`\`\`

### Load Testing

\`\`\`python
import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_test(url, num_requests=1000, concurrency=10):
    """
    Load test the API.

    Args:
        url: API endpoint
        num_requests: total requests to make
        concurrency: number of concurrent requests
    """
    def make_request():
        start = time.time()
        try:
            response = requests.post(
                url,
                json={
                    "data": np.random.randn(1, 3, 224, 224).tolist(),
                    "return_probabilities": False
                },
                timeout=10
            )
            latency = (time.time() - start) * 1000
            return {
                'success': response.status_code == 200,
                'latency': latency
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]

        for future in as_completed(futures):
            results.append(future.result())

    # Analyze results
    successful = [r for r in results if r.get('success')]
    latencies = [r['latency'] for r in successful]

    print(f"\n=== Load Test Results ===")
    print(f"Total requests: {num_requests}")
    print(f"Successful: {len(successful)} ({len(successful)/num_requests*100:.1f}%)")
    print(f"Failed: {num_requests - len(successful)}")
    print(f"\nLatency (ms):")
    print(f"  Mean: {np.mean(latencies):.2f}")
    print(f"  Median: {np.median(latencies):.2f}")
    print(f"  P95: {np.percentile(latencies, 95):.2f}")
    print(f"  P99: {np.percentile(latencies, 99):.2f}")
    print(f"  Max: {np.max(latencies):.2f}")

    return results

# Run load test
load_test('http://localhost:8000/predict', num_requests=1000, concurrency=20)
\`\`\`

### Deployment Checklist

**Pre-deployment:**
- [ ] Export model to TorchScript/ONNX
- [ ] Test exported model accuracy
- [ ] Set up logging and metrics
- [ ] Write comprehensive tests
- [ ] Create health check endpoint
- [ ] Document API endpoints

**Deployment:**
- [ ] Build Docker image
- [ ] Run load tests
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure autoscaling
- [ ] Set up CI/CD pipeline
- [ ] Prepare rollback plan

**Post-deployment:**
- [ ] Monitor metrics (latency, throughput, errors)
- [ ] Set up alerts
- [ ] Monitor resource usage
- [ ] A/B test if updating existing model
- [ ] Document production issues

### Practice Exercises

- [ ] Build complete FastAPI inference server
- [ ] Export model to TorchScript
- [ ] Containerize with Docker
- [ ] Set up Prometheus monitoring
- [ ] Run load tests
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Implement A/B testing
- [ ] Set up automated deployment pipeline`
				}
			]
		}
	]
};

async function seed() {
	console.log('Seeding Deep Learning Part 2 path...');

	const pathResult = db.insert(schema.paths).values({
		name: deepLearningPart2Path.name,
		description: deepLearningPart2Path.description,
		color: deepLearningPart2Path.color,
		language: deepLearningPart2Path.language,
		skills: deepLearningPart2Path.skills,
		startHint: deepLearningPart2Path.startHint,
		difficulty: deepLearningPart2Path.difficulty,
		estimatedWeeks: deepLearningPart2Path.estimatedWeeks,
		schedule: deepLearningPart2Path.schedule
	}).returning().get();

	console.log(`Created path: ${deepLearningPart2Path.name}`);

	for (let i = 0; i < deepLearningPart2Path.modules.length; i++) {
		const mod = deepLearningPart2Path.modules[i];
		const moduleResult = db.insert(schema.modules).values({
			pathId: pathResult.id,
			name: mod.name,
			description: mod.description,
			orderIndex: i
		}).returning().get();

		console.log(`  Created module: ${mod.name}`);

		for (let j = 0; j < mod.tasks.length; j++) {
			const task = mod.tasks[j];
			db.insert(schema.tasks).values({
				moduleId: moduleResult.id,
				title: task.title,
				description: task.description,
				details: task.details,
				orderIndex: j,
				completed: false
			}).run();
		}
		console.log(`    Added ${mod.tasks.length} tasks`);
	}

	console.log('\nSeeding complete!');
}

seed().catch(console.error);
