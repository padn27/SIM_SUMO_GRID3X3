# Treinamento DQN com SUMO-RL

Este projeto implementa o controle de semáforos usando **Aprendizado por Reforço (DQN)** integrado ao **SUMO-RL**.

---

##  Pré-requisitos

### **SUMO**
O SUMO precisa estar instalado no sistema.

### **Variável de ambiente `SUMO_HOME`**
O script `gerar_rotas_central.py` precisa saber onde o SUMO está localizado.  
No terminal, defina a variável (ajuste o caminho conforme sua instalação):

```bash
export SUMO_HOME="/usr/share/sumo"
```

### **Python e dependências**
Instale as bibliotecas necessárias para o projeto:

```bash
pip install torch sumo-rl gymnasium matplotlib
```

---

## 2. Gerar as Rotas

Execute o script **apenas uma vez** para gerar os arquivos de rota:

```bash
python3 gerar_rotas_central.py
```

---

##  3. Configurar os Semáforos

Edite o arquivo:

```
baseSumo/tls_config.json
```

Esse arquivo informa ao script de treino qual tipo de controle cada semáforo utilizará:

| Código | Tipo de Controle | Descrição |
|:------:|:------------------|:------------|
| `"R"` | **Reinforcement Learning Agent** | Controlado pela IA |
| `"F"` | **Fixo** | Alterna fases em ordem e intervalos fixos |

---

##  4: Rodar o Treinamento

Execute o script `treinar_dqn_sumorl.py` com os argumentos desejados:

### **Argumentos disponíveis**

| Argumento | Padrão | Descrição |
|:-----------|:--------|:-----------|
| `--validacao` | *(nenhum)* | Se presente, abre a GUI do SUMO durante a simulação |
| `--episodios` | `100` | Número total de episódios de treinamento |
| `--rotasdir` | `rotas_jtr` | Pasta onde estão os arquivos de rota utilizados |
| `--troca` | `10` | A cada quantos episódios o script troca o arquivo de tráfego |
| `--net` | `baseSumo/grid.net.xml` | Caminho para o arquivo do mapa (`.net.xml`) |
| `--add` | `baseSumo/grid.add.xml` | Caminho para o arquivo adicional (`.add.xml`) |
| `--tls` | `baseSumo/tls_config.json` | Caminho para o JSON que define o tipo de cada semáforo |

> A divisão `episodios / troca` deve ser igual ao número de arquivos de rota dentro da pasta `rotas_jtr`.

---

### **Exemplo de uso**

```bash
python3 treinar_dqn_sumorl.py --validacao --episodios 8100 --rotasdir rotas_jtr --net baseSumo/grid.net.xml --add baseSumo/grid.add.xml --tls baseSumo/tls_config.json --troca 100
```

> Treina os agentes configurados como `"R"` no arquivo `tls_config.json` por **8100 episódios**,  
> trocando o arquivo de tráfego a cada **100 episódios**, mantendo a memória do treinamento.

---

## Configurações da Simulação

O ambiente é criado no código `treinar_dqn_sumorl.py` com a seguinte estrutura:

```python
env = sumo_rl.parallel_env(
    net_file=NET_FILE,
    route_file=nova_rota,
    use_gui=USE_GUI,
    num_seconds=3600,
    delta_time=10,
    reward_fn='diff-waiting-time'
)
```

### **Parâmetros principais**

| Parâmetro | Descrição |
|:-----------|:-----------|
| `num_seconds` | Duração total (em segundos) de cada episódio, com referência ao SUMO |
| `delta_time` | Quantos passos do SUMO equivalem a 1 step de aprendizado |
| `reward_fn` | Função de recompensa usada pelo agente (ex: `'diff-waiting-time'` ou `'minha_recompensa'`) |

Pode-se alterar `reward_fn` no código para usar funções personalizadas.

---
