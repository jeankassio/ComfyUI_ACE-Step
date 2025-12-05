#!/usr/bin/env python
"""
Script de teste rápido para validar Windows Compatibility.

Execute este script para verificar se as mudanças de compatibilidade do Windows foram aplicadas corretamente.
"""

import sys
import os

def test_imports():
    """Teste 1: Verificar se os novos módulos podem ser importados."""
    print("\n" + "="*60)
    print("TESTE 1: Importacao dos Modulos")
    print("="*60)
    
    try:
        print("\n[1.1] Testando import de audio_utils...")
        from ace_step.audio_utils import (
            load_audio_safe,
            load_audio_safe_stereo,
            load_audio_safe_mono,
            get_audio_info
        )
        print("     [OK] audio_utils importado com sucesso")
        
        print("\n[1.2] Testando import de torch_utils...")
        from ace_step.torch_utils import (
            safe_torch_compile,
            safe_cuda_empty_cache,
            safe_cuda_synchronize,
            get_optimal_device,
            get_optimal_dtype,
            setup_torch_backends
        )
        print("     [OK] torch_utils importado com sucesso")
        
        return True
    except ImportError as e:
        print(f"     [ERRO] Erro ao importar: {e}")
        return False


def test_torch_setup():
    """Teste 2: Verificar configuração do PyTorch."""
    print("\n" + "="*60)
    print("TESTE 2: Setup do PyTorch")
    print("="*60)
    
    try:
        import torch
        from ace_step.torch_utils import setup_torch_backends, get_optimal_device, get_optimal_dtype
        
        print("\n[2.1] Testando setup_torch_backends()...")
        setup_torch_backends()
        print("     [OK] PyTorch configurado com sucesso")
        
        print("\n[2.2] Testando get_optimal_device()...")
        device = get_optimal_device()
        print(f"     [OK] Dispositivo otimo: {device}")
        
        print("\n[2.3] Testando get_optimal_dtype()...")
        dtype = get_optimal_dtype()
        print(f"     [OK] Dtype otimo: {dtype}")
        
        return True
    except Exception as e:
        print(f"     [ERRO] Erro: {e}")
        return False


def test_safe_cuda():
    """Teste 3: Verificar operações CUDA seguras."""
    print("\n" + "="*60)
    print("TESTE 3: Operacoes CUDA Seguras")
    print("="*60)
    
    try:
        import torch
        from ace_step.torch_utils import safe_cuda_empty_cache, safe_cuda_synchronize
        
        print("\n[3.1] Testando safe_cuda_empty_cache()...")
        safe_cuda_empty_cache()
        print("     [OK] Cache CUDA limpo com seguranca")
        
        print("\n[3.2] Testando safe_cuda_synchronize()...")
        safe_cuda_synchronize()
        print("     [OK] CUDA sincronizado com seguranca")
        
        if torch.cuda.is_available():
            print("\n[3.3] GPU disponivel: Sim")
            print(f"     Dispositivo: {torch.cuda.get_device_name(0)}")
            print(f"     CUDA Version: {torch.version.cuda}")
        else:
            print("\n[3.3] GPU disponivel: Nao (usando CPU)")
        
        return True
    except Exception as e:
        print(f"     [ERRO] Erro: {e}")
        return False


def test_safe_compile():
    """Teste 4: Verificar compilação segura."""
    print("\n" + "="*60)
    print("TESTE 4: Compilacao Segura de Modelos")
    print("="*60)
    
    try:
        import torch
        from ace_step.torch_utils import safe_torch_compile, get_optimal_device
        
        print("\n[4.1] Criando modelo simples...")
        device = get_optimal_device()
        # Use float32 to avoid bfloat16 issues with triton on Windows
        dtype = torch.float32
        
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        ).to(device).to(dtype)
        print("     [OK] Modelo criado")
        
        print("\n[4.2] Testando safe_torch_compile()...")
        compiled_model = safe_torch_compile(model, enable_compile=True)
        print("     [OK] Modelo compilado (ou original se compilacao falhar)")
        
        print("\n[4.3] Testando forward pass...")
        x = torch.randn(2, 10, device=device, dtype=dtype)
        with torch.no_grad():
            y = compiled_model(x)
        print(f"     [OK] Forward pass OK: input={x.shape}, output={y.shape}")
        
        return True
    except Exception as e:
        print(f"     [ERRO] Erro: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_files_modified():
    """Teste 5: Verificar se arquivos foram modificados corretamente."""
    print("\n" + "="*60)
    print("TESTE 5: Verificacao de Arquivos Modificados")
    print("="*60)
    
    files_to_check = {
        "ace_step_nodes.py": ["safe_torch_compile", "setup_torch_backends"],
        "ace_step/pipeline_ace_step.py": ["safe_cuda_empty_cache"],
        "ace_step/cpu_offload.py": ["safe_cuda_empty_cache", "safe_cuda_synchronize"],
        "ace_step/music_dcae/music_dcae_pipeline.py": ["load_audio_safe_stereo"],
        "ace_step/text2music_dataset.py": ["load_audio_safe_stereo"],
    }
    
    all_ok = True
    
    for filepath, required_strings in files_to_check.items():
        print(f"\n[5.x] Verificando {filepath}...")
        
        full_path = os.path.join(os.getcwd(), filepath)
        
        if not os.path.exists(full_path):
            print(f"     [ERRO] Arquivo nao encontrado: {filepath}")
            all_ok = False
            continue
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            missing = []
            for req_string in required_strings:
                if req_string not in content:
                    missing.append(req_string)
            
            if missing:
                print(f"     [ERRO] Faltando strings esperadas: {missing}")
                all_ok = False
            else:
                print(f"     [OK] Arquivo contem todas as mudancas esperadas")
        
        except Exception as e:
            print(f"     [ERRO] Erro ao ler arquivo: {e}")
            all_ok = False
    
    return all_ok


def print_summary(results):
    """Imprime resumo dos testes."""
    print("\n" + "="*60)
    print("RESUMO DOS TESTES")
    print("="*60)
    
    test_names = [
        "Importacao de Modulos",
        "Setup do PyTorch",
        "Operacoes CUDA Seguras",
        "Compilacao Segura",
        "Verificacao de Arquivos"
    ]
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nResultados: {passed}/{total} testes passaram\n")
    
    for i, (name, result) in enumerate(zip(test_names, results), 1):
        status = "[OK]" if result else "[FALHOU]"
        print(f"  Teste {i}: {name:<35} {status}")
    
    print("\n" + "="*60)
    
    if passed == total:
        print("[OK] TODOS OS TESTES PASSARAM!")
        print("\nWindows Compatibility foi implementado com sucesso!")
        return 0
    else:
        print(f"[AVISO] {total - passed} teste(s) falharam!")
        print("\nPor favor, revise os erros acima.")
        return 1


def main():
    """Função principal."""
    print("\n")
    print("=" * 60)
    print("Windows Compatibility Test Suite")
    print("=" * 60)
    
    # Mudar para o diretório correto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) != "ComfyUI_ACE-Step":
        os.chdir(script_dir)
    
    results = []
    
    try:
        results.append(test_imports())
        results.append(test_torch_setup())
        results.append(test_safe_cuda())
        results.append(test_safe_compile())
        results.append(test_files_modified())
    except Exception as e:
        print(f"\nERRO critico durante testes: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
