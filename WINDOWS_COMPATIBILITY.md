## Windows Compatibility Updates - ACE-Step

Este documento descreve as mudanças realizadas para melhorar a compatibilidade com Windows, especialmente com CUDA.

### Problemas Resolvidos

1. **Incompatibilidade do torchcodec no Windows com CUDA**
   - `torchaudio.load()` pode falhar no Windows quando usa torchcodec com CUDA
   - Isso ocorre porque torchcodec não tem suporte completo para Windows + CUDA

2. **Problemas com torch.compile() no Windows**
   - `torch.compile()` pode falhar ou não estar disponível em certas configurações do Windows
   - Triton (dependência do torch.compile) pode não funcionar bem no Windows

3. **Chamadas diretas a CUDA sem verificação**
   - `torch.cuda.empty_cache()` e `torch.cuda.synchronize()` podem causar erros se não houver GPU disponível

### Soluções Implementadas

#### 1. Novo módulo: `ace_step/audio_utils.py`

Fornece funções seguras para carregar áudio com fallbacks automáticos:

```python
from ace_step.audio_utils import load_audio_safe_stereo

# Carrega áudio com fallback automático para librosa se torchaudio falhar
audio, sr = load_audio_safe_stereo("audio.wav")
```

**Funções disponíveis:**
- `load_audio_safe()`: Carrega áudio com fallback para librosa
- `load_audio_safe_stereo()`: Garante saída em estéreo
- `load_audio_safe_mono()`: Garante saída em mono
- `get_audio_info()`: Obtém informações do áudio sem carregar tudo

**Como funciona:**
1. Tenta usar `torchaudio.load()` (mais rápido)
2. Se falhar, tenta `librosa.load()` (mais compatível)
3. Reasmostra automaticamente se necessário
4. Avisa o usuário qual backend foi usado

#### 2. Novo módulo: `ace_step/torch_utils.py`

Utilitários PyTorch com compatibilidade para Windows:

```python
from ace_step.torch_utils import safe_torch_compile, safe_cuda_empty_cache

# Compila modelo com fallback seguro
model = safe_torch_compile(model, enable_compile=True)

# Limpa cache CUDA de forma segura
safe_cuda_empty_cache()
```

**Funções disponíveis:**
- `safe_torch_compile()`: Compila modelo com fallback gracioso
- `safe_cuda_empty_cache()`: Limpa CUDA cache com segurança
- `safe_cuda_synchronize()`: Sincroniza CUDA com segurança
- `get_optimal_device()`: Detecta melhor dispositivo (CUDA/MPS/CPU)
- `get_optimal_dtype()`: Recomenda dtype ótimo para o dispositivo
- `setup_torch_backends()`: Configura backends PyTorch com segurança

#### 3. Arquivos Atualizados

**`ace_step_nodes.py`**
- Adicionado import de `torch_utils`
- Chamada a `setup_torch_backends()` na inicialização
- Substituído `torch.compile()` por `safe_torch_compile()`

**`ace_step/pipeline_ace_step.py`**
- Adicionado import de `safe_cuda_empty_cache`
- Substituído `torch.cuda.empty_cache()` por `safe_cuda_empty_cache()`

**`ace_step/cpu_offload.py`**
- Adicionado import de `torch_utils`
- Substituído `torch.cuda.empty_cache()` e `torch.cuda.synchronize()` por funções seguras

**`ace_step/music_dcae/music_dcae_pipeline.py`**
- Adicionado import de `load_audio_safe_stereo`
- Substituído `torchaudio.load()` por `load_audio_safe_stereo()` em todas as ocorrências

**`ace_step/text2music_dataset.py`**
- Adicionado import de `load_audio_safe_stereo`
- Substituído `torchaudio.load()` por `load_audio_safe_stereo()` na função `get_audio()`

### Comportamento em Diferentes Sistemas

#### Windows com CUDA
- ✅ Audio loading com fallback librosa se torchaudio falhar
- ✅ torch.compile() tentará compilar, mas continua funcionando se falhar
- ✅ Chamadas CUDA são seguras e verificadas

#### Windows sem GPU
- ✅ Usa CPU automaticamente
- ✅ Audio loading funciona com librosa
- ✅ Sem erros ao tentar acessar CUDA

#### Linux/Mac com CUDA/MPS
- ✅ Funciona normalmente com todas as otimizações
- ✅ Audio loading usa torchaudio (mais rápido)
- ✅ torch.compile() funciona quando disponível

### Dependências

O código já incluía `librosa` no `requirements.txt`. Nenhuma dependência nova foi adicionada.

### Testes Recomendados

1. **Testar carregamento de áudio em Windows**
   ```python
   from ace_step.audio_utils import load_audio_safe_stereo
   audio, sr = load_audio_safe_stereo("test.wav")
   print(f"Carregado: {audio.shape}, SR: {sr}")
   ```

2. **Testar compilação de modelo**
   ```python
   from ace_step.torch_utils import safe_torch_compile
   model = MyModel()
   model = safe_torch_compile(model)
   # Deve funcionar sem erros
   ```

3. **Testar limpeza de CUDA**
   ```python
   from ace_step.torch_utils import safe_cuda_empty_cache
   safe_cuda_empty_cache()
   # Deve ser seguro em qualquer plataforma
   ```

### Notas Importantes

- O código continua funcionando em Linux/Mac sem alterações de comportamento
- Avisos são exibidos quando fallbacks são usados, mas o código continua funcionando
- Performance pode ser ligeiramente reduzida no Windows ao usar librosa, mas garante funcionamento
- torch.compile() agora é graciosamente desabilitado se não funcionar, em vez de causar crash

### Compatibilidade Forward

As mudanças são totalmente backwards-compatible:
- Código existente não precisa de alterações
- Novos recursos continuam funcionando como esperado
- Fallbacks são automáticos e transparentes
