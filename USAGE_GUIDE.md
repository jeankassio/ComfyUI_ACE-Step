# Guia de Uso - Windows Compatibility Fallbacks

## 🎯 Visão Geral

O código ACE-Step foi otimizado para funcionar perfeitamente no Windows com CUDA, resolvendo problemas com `torchcodec` que não é compatível com Windows.

## 📖 Como Usar

### 1. Carregamento de Áudio (Load)

#### Uso Básico:
```python
from ace_step.audio_utils import load_audio_safe_stereo

# Carrega áudio em estéreo (automático fallback para librosa se necessário)
audio, sample_rate = load_audio_safe_stereo("music.wav")
# audio shape: (2, num_samples)
# sample_rate: int (ex: 44100)
```

#### Variações:
```python
from ace_step.audio_utils import load_audio_safe, load_audio_safe_mono

# Carregamento genérico
audio, sr = load_audio_safe("music.wav")

# Carregamento em mono
audio_mono, sr = load_audio_safe_mono("music.wav")

# Com resampling
audio, sr = load_audio_safe_stereo("music.wav", sr=44100)
```

### 2. Salvamento de Áudio (Save)

#### Uso Básico:
```python
from ace_step.audio_utils import save_audio_safe

# Salva áudio com fallback automático
save_audio_safe("output.wav", audio_tensor, 44100)
# Tenta: torchaudio.save() → soundfile.write() → librosa.output.write_wav()
```

#### Salvamento em Batch:
```python
from ace_step.audio_utils import save_audio_safe_batch

# Salvar múltiplos arquivos
results = save_audio_safe_batch(
    ["out1.wav", "out2.wav", "out3.wav"],
    [audio1, audio2, audio3],
    44100  # ou [44100, 48000, 44100] para sample rates diferentes
)
# returns: [True, True, False] indicando sucesso/falha
```

### 3. Informações de Áudio

```python
from ace_step.audio_utils import get_audio_info

info = get_audio_info("music.wav")
# Retorna: {
#     'sample_rate': 44100,
#     'num_frames': 2048000,
#     'num_channels': 2
# }
```

## 🔄 Fluxos de Trabalho Comuns

### Workflow 1: Processar Áudio
```python
from ace_step.audio_utils import load_audio_safe_stereo, save_audio_safe
import torch

# 1. Carregar
audio, sr = load_audio_safe_stereo("input.wav")
print(f"Carregado: {audio.shape}, SR: {sr}")

# 2. Processar (exemplo)
audio_processed = audio * 0.9  # Reduzir volume
audio_processed = torch.clamp(audio_processed, -1, 1)

# 3. Salvar
save_audio_safe("output.wav", audio_processed, sr)
print("Salvo com sucesso!")
```

### Workflow 2: Usar em Dataset
```python
from ace_step.audio_utils import load_audio_safe_stereo

class MyDataset:
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        # Funciona em Windows com fallback automático!
        audio, sr = load_audio_safe_stereo(audio_path)
        return audio, sr
```

### Workflow 3: Usar em Nodes ComfyUI
```python
from ace_step.audio_utils import save_audio_safe

class MyNode:
    def execute(self, audio_tensor, sample_rate):
        # Processa áudio
        output_audio = self.process(audio_tensor)
        
        # Salva com fallback automático
        output_path = "/tmp/output.wav"
        save_audio_safe(output_path, output_audio, sample_rate)
        
        return output_path
```

## 🔧 Tratamento de Erros

### Capturar Erros:
```python
from ace_step.audio_utils import load_audio_safe_stereo, save_audio_safe

try:
    audio, sr = load_audio_safe_stereo("audio.wav")
except RuntimeError as e:
    print(f"Erro ao carregar: {e}")
    # Ambos torchaudio E librosa falharam

try:
    save_audio_safe("output.wav", audio, sr)
except RuntimeError as e:
    print(f"Erro ao salvar: {e}")
    # Todos os backends falharam
```

### Logging:
O código exibe mensagens informativas:
```
[AudioUtils] torchaudio.load failed: [specific error]
[AudioUtils] Falling back to librosa for Windows compatibility...
[AudioUtils] Successfully saved audio to output.wav using soundfile
```

## 💡 Dicas de Performance

1. **Windows**: Primeira execução pode ser lenta (librosa carrega modelo), execuções subsequentes são mais rápidas
2. **Linux**: Usa torchaudio nativo, performance máxima
3. **Batch Operations**: Use `save_audio_safe_batch()` para salvar vários arquivos

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'soundfile'"
```bash
pip install soundfile
# Ou já incluído em requirements.txt
```

### "ModuleNotFoundError: No module named 'librosa'"
```bash
pip install librosa
# Ou já incluído em requirements.txt
```

### "RuntimeError: Failed to load audio..."
- Arquivo de áudio não existe ou está corrompido
- Formato não suportado (tente converter para WAV)
- Caminho com caracteres especiais/unicode

### Áudio salvou mas está silencioso
- Verifique se `audio_tensor` não está em GPU (deve estar em CPU antes de salvar)
- Verifique dtype (deve ser float32)
- Verifique range de valores (deve estar entre -1 e 1 para máxima qualidade)

```python
# Garantir formato correto antes de salvar
audio = audio.cpu().float()
audio = torch.clamp(audio, -1, 1)  # Normalizador se necessário
save_audio_safe("output.wav", audio, sr)
```

## 📊 Compatibilidade Garantida

| Plataforma | Status | Método |
|-----------|--------|--------|
| Windows + CUDA | ✅ | librosa (fallback) |
| Windows + CPU | ✅ | librosa (fallback) |
| Linux + CUDA | ✅ | torchaudio (nativo) |
| Mac + MPS | ✅ | torchaudio (nativo) |

## 🔍 Verificar Sistema

```python
import torch
import platform

print(f"Plataforma: {platform.system()}")
print(f"CUDA Disponível: {torch.cuda.is_available()}")
print(f"Dispositivo: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

## 📚 Exemplos Completos

### Exemplo 1: Simples
```python
from ace_step.audio_utils import load_audio_safe_stereo, save_audio_safe

audio, sr = load_audio_safe_stereo("input.wav")
save_audio_safe("output.wav", audio, sr)
```

### Exemplo 2: Com Processamento
```python
from ace_step.audio_utils import load_audio_safe_stereo, save_audio_safe
import torch

audio, sr = load_audio_safe_stereo("input.wav")
# Aplicar filtro simples
audio_filtered = torch.nn.functional.max_pool1d(audio.unsqueeze(0), 3, 1, padding=1)
audio_filtered = audio_filtered.squeeze(0)
save_audio_safe("output.wav", audio_filtered, sr)
```

### Exemplo 3: Batch Processing
```python
from ace_step.audio_utils import load_audio_safe_stereo, save_audio_save_batch
import glob

audio_files = glob.glob("input/*.wav")
audios = []
for f in audio_files:
    audio, sr = load_audio_safe_stereo(f)
    audios.append(audio * 0.5)  # Reduzir volume

output_paths = [f.replace("input", "output") for f in audio_files]
save_audio_safe_batch(output_paths, audios, sr)
```

---

**Documentação Completa!**  
Para mais detalhes, veja `WINDOWS_TORCHCODEC_FALLBACK.md`
