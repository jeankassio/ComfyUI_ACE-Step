"""
Exemplos de uso dos novos módulos de compatibilidade com Windows.

Este arquivo demonstra como usar os novos utilitários de compatibilidade.
"""

import torch
import sys
import os

# Adicionar o caminho do módulo
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from ace_step.audio_utils import (
    load_audio_safe,
    load_audio_safe_stereo,
    load_audio_safe_mono,
    get_audio_info
)
from ace_step.torch_utils import (
    safe_torch_compile,
    safe_cuda_empty_cache,
    safe_cuda_synchronize,
    get_optimal_device,
    get_optimal_dtype,
    setup_torch_backends
)


def example_audio_loading():
    """
    Exemplo 1: Carregamento de áudio com fallback automático.
    """
    print("=" * 60)
    print("Exemplo 1: Carregamento de Áudio")
    print("=" * 60)
    
    # Este exemplo assume que você tem um arquivo de áudio
    audio_path = "seu_audio.wav"
    
    if not os.path.exists(audio_path):
        print(f"Arquivo de exemplo não encontrado: {audio_path}")
        print("Criando exemplo hipotético...")
        return
    
    # Carregamento simples
    print("\n1. Carregamento simples:")
    audio, sr = load_audio_safe(audio_path)
    print(f"   Áudio carregado: shape={audio.shape}, sample_rate={sr}")
    
    # Carregamento garantindo estéreo
    print("\n2. Carregamento em estéreo:")
    audio_stereo, sr = load_audio_safe_stereo(audio_path)
    print(f"   Áudio estéreo: shape={audio_stereo.shape}, sample_rate={sr}")
    
    # Carregamento garantindo mono
    print("\n3. Carregamento em mono:")
    audio_mono, sr = load_audio_safe_mono(audio_path)
    print(f"   Áudio mono: shape={audio_mono.shape}, sample_rate={sr}")
    
    # Carregamento com resampling
    print("\n4. Carregamento com resampling para 16kHz:")
    audio_16k, sr_16k = load_audio_safe(audio_path, sr=16000)
    print(f"   Áudio: shape={audio_16k.shape}, sample_rate={sr_16k}")
    
    # Informações do arquivo
    print("\n5. Informações do arquivo de áudio:")
    info = get_audio_info(audio_path)
    print(f"   Sample rate: {info['sample_rate']}")
    print(f"   Número de frames: {info['num_frames']}")
    print(f"   Número de canais: {info['num_channels']}")


def example_torch_compilation():
    """
    Exemplo 2: Compilação segura de modelos.
    """
    print("\n" + "=" * 60)
    print("Exemplo 2: Compilação Segura de Modelos")
    print("=" * 60)
    
    # Criar um modelo simples
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    print(f"\nModelo original: {type(model)}")
    
    # Compilar modelo com fallback seguro
    print("\nTentando compilar modelo (com fallback seguro)...")
    compiled_model = safe_torch_compile(model, enable_compile=True)
    print(f"Modelo após compilação: {type(compiled_model)}")
    
    # Testar o modelo
    x = torch.randn(1, 10)
    y = compiled_model(x)
    print(f"Teste: input shape={x.shape}, output shape={y.shape}")


def example_torch_setup():
    """
    Exemplo 3: Setup de PyTorch com compatibilidade.
    """
    print("\n" + "=" * 60)
    print("Exemplo 3: Setup de PyTorch")
    print("=" * 60)
    
    # Setup automático
    print("\nConfigurando PyTorch backends...")
    setup_torch_backends()
    print("Setup concluído!")
    
    # Obter informações do dispositivo
    device = get_optimal_device()
    dtype = get_optimal_dtype()
    
    print(f"\nDispositivo ótimo: {device}")
    print(f"Dtype ótimo: {dtype}")
    
    # Criar tensor e mover para dispositivo
    x = torch.randn(10, 10, dtype=dtype)
    x = x.to(device)
    print(f"Tensor criado: device={x.device}, dtype={x.dtype}")


def example_safe_cuda_operations():
    """
    Exemplo 4: Operações CUDA seguras.
    """
    print("\n" + "=" * 60)
    print("Exemplo 4: Operações CUDA Seguras")
    print("=" * 60)
    
    # Operações CUDA seguras (funcionam em CPU também)
    print("\nRealizando operações CUDA seguras...")
    
    safe_cuda_empty_cache()
    print("✓ Cache CUDA limpo com segurança")
    
    safe_cuda_synchronize()
    print("✓ CUDA sincronizado com segurança")
    
    print("\nTodas as operações completadas com sucesso!")
    print("(Nota: Se CUDA não estiver disponível, as operações foram ignoradas com segurança)")


def example_end_to_end():
    """
    Exemplo 5: Fluxo completo de processamento.
    """
    print("\n" + "=" * 60)
    print("Exemplo 5: Fluxo Completo de Processamento")
    print("=" * 60)
    
    # Setup
    print("\n1. Setup de PyTorch...")
    setup_torch_backends()
    print("   ✓ Concluído")
    
    # Obter dispositivo ótimo
    device = get_optimal_device()
    dtype = get_optimal_dtype()
    print(f"\n2. Dispositivo: {device}, Dtype: {dtype}")
    
    # Criar modelo
    print("\n3. Criando modelo...")
    model = torch.nn.Linear(100, 50).to(device).to(dtype)
    print("   ✓ Modelo criado")
    
    # Compilar modelo
    print("\n4. Compilando modelo com fallback...")
    model = safe_torch_compile(model, enable_compile=True)
    print("   ✓ Modelo compilado (ou original se compilação falhar)")
    
    # Processar dados
    print("\n5. Processando dados...")
    x = torch.randn(10, 100, device=device, dtype=dtype)
    with torch.no_grad():
        y = model(x)
    print(f"   ✓ Dados processados: input={x.shape}, output={y.shape}")
    
    # Limpeza
    print("\n6. Limpeza de recursos...")
    safe_cuda_empty_cache()
    print("   ✓ Cache limpo")
    
    print("\nFluxo completo finalizado com sucesso!")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "EXEMPLOS DE USO - ACE-Step Windows" + " " * 11 + "║")
    print("║" + " " * 14 + "Compatibilidade com CUDA no Windows" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Executar exemplos
    try:
        example_torch_setup()
        example_torch_compilation()
        example_safe_cuda_operations()
        example_end_to_end()
        
        # Audio loading é opcional pois requer arquivo
        print("\n" + "=" * 60)
        print("Nota sobre Exemplo de Áudio:")
        print("=" * 60)
        print("O exemplo de carregamento de áudio não foi executado pois")
        print("requer um arquivo de áudio real. Para testá-lo:")
        print("\n  from ace_step.audio_utils import load_audio_safe_stereo")
        print("  audio, sr = load_audio_safe_stereo('seu_arquivo.wav')")
        print("\nO código funcionará em Windows, Linux e Mac,")
        print("com fallback automático para librosa se necessário.")
        
    except Exception as e:
        print(f"\nErro durante execução dos exemplos: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Exemplos concluídos!")
    print("=" * 60 + "\n")
