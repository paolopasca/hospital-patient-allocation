"""
Modelo MILP (Mixed Integer Linear Programming) para o problema de alocação de pacientes.
Usa o Gurobi como solver para encontrar a solução ótima (Método 1 - Branch & Bound).

VERSÃO CORRIGIDA - Problemas resolvidos:
1. Restrição de capacidade de camas corrigida (uso correto de quicksum)
2. Restrição de workload corrigida (separação de constantes e variáveis)
3. Melhor estruturação das expressões lineares
"""

import gurobipy as gp
from gurobipy import GRB
import time
from data_parser import PatientAllocationData


class PatientAllocationMILP:
    """Modelo MILP para alocação de pacientes em hospitais."""
    
    def __init__(self, data: PatientAllocationData, lambda1=0.5, lambda2=0.5):
        """
        Inicializa o modelo MILP.
        
        Args:
            data: Objeto com os dados do problema
            lambda1: Peso do objetivo 1 (custo operacional)
            lambda2: Peso do objetivo 2 (equilíbrio de carga)
        """
        self.data = data
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # Criar modelo Gurobi
        self.model = gp.Model("PatientAllocation")
        
        # Variáveis de decisão
        self.y = {}  # y[p,w,d] = 1 se paciente p é admitido na enfermaria w no dia d
        self.x = {}  # x[w,d] = carga de trabalho normalizada na enfermaria w no dia d
        self.z = None  # z = máximo da carga de trabalho (para minimizar)
        self.v_overtime = {}   # v[s,d] = overtime da especialização s no dia d
        self.u_undertime = {}  # u[s,d] = undertime da especialização s no dia d
        
        # Variáveis auxiliares
        self.solution = None
        self.objective_value = None
        self.solve_time = None
        
    def build_model(self):
        """Constrói o modelo completo com variáveis, restrições e função objetivo."""
        print("Construindo o modelo MILP...")
        
        # 1. CRIAR VARIÁVEIS DE DECISÃO
        self._create_variables()
        
        # 2. ADICIONAR RESTRIÇÕES
        self._add_constraints()
        
        # 3. DEFINIR FUNÇÃO OBJETIVO
        self._set_objective()
        
        print("✓ Modelo construído com sucesso!")
        print(f"  - Variáveis: {self.model.NumVars}")
        print(f"  - Restrições: {self.model.NumConstrs}")
    
    def _create_variables(self):
        """Cria todas as variáveis de decisão do modelo."""
        
        # Y[p,w,d] - Variável binária de alocação
        print("  Criando variáveis Y (alocação)...")
        for patient_id, patient in self.data.patients.items():
            spec = patient['specialization']
            earliest = patient['earliest']
            latest = patient['latest']
            
            for ward_name, ward in self.data.wards.items():
                # Só criar variável se a enfermaria aceitar a especialização
                if (spec == ward['major_specialization'] or 
                    spec in ward['minor_specializations']):
                    
                    for d in range(earliest, min(latest + 1, self.data.num_days)):
                        self.y[patient_id, ward_name, d] = self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f"y_{patient_id}_{ward_name}_{d}"
                        )
        
        # X[w,d] - Carga de trabalho normalizada
        print("  Criando variáveis X (carga de trabalho)...")
        for ward_name in self.data.wards.keys():
            for d in range(self.data.num_days):
                self.x[ward_name, d] = self.model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    name=f"x_{ward_name}_{d}"
                )
        
        # Z - Máximo da carga de trabalho
        print("  Criando variável Z (máximo)...")
        self.z = self.model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="z_max_workload"
        )
        
        # V[s,d] - Overtime
        print("  Criando variáveis V (overtime)...")
        for spec in self.data.specialisms.keys():
            for d in range(self.data.num_days):
                self.v_overtime[spec, d] = self.model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    name=f"v_overtime_{spec}_{d}"
                )
        
        # U[s,d] - Undertime
        print("  Criando variáveis U (undertime)...")
        for spec in self.data.specialisms.keys():
            for d in range(self.data.num_days):
                self.u_undertime[spec, d] = self.model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    name=f"u_undertime_{spec}_{d}"
                )
    
    def _add_constraints(self):
        """Adiciona todas as restrições do modelo."""
        
        # RESTRIÇÃO 1: Cada paciente deve ser admitido exatamente uma vez
        print("  Adicionando restrição: cada paciente admitido uma vez...")
        for patient_id in self.data.patients.keys():
            self.model.addConstr(
                gp.quicksum(self.y[key] for key in self.y.keys() if key[0] == patient_id) == 1,
                name=f"admit_once_{patient_id}"
            )
        
        # RESTRIÇÃO 2: Capacidade de camas das enfermarias [CORRIGIDA]
        print("  Adicionando restrição: capacidade de camas...")
        for ward_name, ward in self.data.wards.items():
            for d in range(self.data.num_days):
                # Somar APENAS as variáveis de decisão
                patients_in_ward = gp.quicksum(
                    self.y[patient_id, ward_name, admit_day]
                    for patient_id, patient in self.data.patients.items()
                    for admit_day in range(max(0, d - patient['los'] + 1), min(d + 1, self.data.num_days))
                    if (patient_id, ward_name, admit_day) in self.y
                    and admit_day <= d < admit_day + patient['los']
                )
                
                # Constraint: carryover (constante) + novas admissões (variáveis) <= capacidade
                self.model.addConstr(
                    ward['carryover_patients'][d] + patients_in_ward <= ward['bed_capacity'],
                    name=f"bed_capacity_{ward_name}_{d}"
                )
        
        # RESTRIÇÃO 3: Tempo do bloco operatório (OT)
        print("  Adicionando restrição: tempo do bloco operatório...")
        for spec in self.data.specialisms.keys():
            for d in range(self.data.num_days):
                # Tempo total usado = soma das durações de cirurgia dos pacientes admitidos no dia d
                ot_used = gp.quicksum(
                    self.data.patients[patient_id]['surgery_duration'] * self.y[patient_id, ward_name, admit_day]
                    for (patient_id, ward_name, admit_day) in self.y.keys()
                    if admit_day == d and self.data.patients[patient_id]['specialization'] == spec
                )
                
                ot_available = self.data.specialisms[spec]['ot_time'][d]
                
                # ot_used + u - v = ot_available
                self.model.addConstr(
                    ot_used + self.u_undertime[spec, d] - self.v_overtime[spec, d] == ot_available,
                    name=f"ot_time_{spec}_{d}"
                )
        
        # RESTRIÇÃO 4: Cálculo da carga de trabalho normalizada X[w,d] [CORRIGIDA]
        print("  Adicionando restrição: cálculo de carga de trabalho...")
        for ward_name, ward in self.data.wards.items():
            workload_capacity = ward['workload_capacity']
            
            for d in range(self.data.num_days):
                # Carga de trabalho das NOVAS admissões (só variáveis)
                workload_from_new_patients = 0
                
                for patient_id, patient in self.data.patients.items():
                    spec = patient['specialization']
                    los = patient['los']
                    workload_per_day = patient['workload_per_day']
                    
                    # Fator de escala se for especialização menor
                    if spec != ward['major_specialization'] and spec in ward['minor_specializations']:
                        scaling_factor = self.data.specialisms[spec]['workload_factor']
                    else:
                        scaling_factor = 1.0
                    
                    # Para cada dia de admissão possível
                    for admit_day in range(self.data.num_days):
                        if (patient_id, ward_name, admit_day) in self.y:
                            # Verificar se paciente está internado no dia d
                            if admit_day <= d < admit_day + los:
                                day_of_stay = d - admit_day
                                if day_of_stay < len(workload_per_day):
                                    workload_contribution = workload_per_day[day_of_stay] * scaling_factor
                                    workload_from_new_patients += workload_contribution * self.y[patient_id, ward_name, admit_day]
                
                # X[w,d] * capacity = carryover (constante) + new_patients (variáveis)
                self.model.addConstr(
                    self.x[ward_name, d] * workload_capacity == ward['carryover_workload'][d] + workload_from_new_patients,
                    name=f"workload_{ward_name}_{d}"
                )
        
        # RESTRIÇÃO 5: Z >= X[w,d] para todos w,d (Z é o máximo)
        print("  Adicionando restrição: Z é o máximo da carga...")
        for ward_name in self.data.wards.keys():
            for d in range(self.data.num_days):
                self.model.addConstr(
                    self.z >= self.x[ward_name, d],
                    name=f"z_max_{ward_name}_{d}"
                )
    
    def _set_objective(self):
        """Define a função objetivo (combinação dos dois objetivos)."""
        print("  Definindo função objetivo...")
        
        # OBJETIVO 1: Custo operacional (overtime + undertime + delays)
        f1_overtime = self.data.weight_overtime * gp.quicksum(
            self.v_overtime[spec, d]
            for spec in self.data.specialisms.keys()
            for d in range(self.data.num_days)
        )
        
        f1_undertime = self.data.weight_undertime * gp.quicksum(
            self.u_undertime[spec, d]
            for spec in self.data.specialisms.keys()
            for d in range(self.data.num_days)
        )
        
        f1_delays = self.data.weight_delay * gp.quicksum(
            (d - self.data.patients[patient_id]['earliest']) * self.y[patient_id, ward_name, d]
            for (patient_id, ward_name, d) in self.y.keys()
        )
        
        f1 = f1_overtime + f1_undertime + f1_delays
        
        # OBJETIVO 2: Equilíbrio de carga (minimizar o máximo)
        f2 = self.z
        
        # OBJETIVO COMBINADO
        objective = self.lambda1 * f1 + self.lambda2 * f2
        
        self.model.setObjective(objective, GRB.MINIMIZE)
        
        print(f"  ✓ Objetivo definido: {self.lambda1}*f1 + {self.lambda2}*f2")
    
    def solve(self, time_limit=600, threads=4, verbose=True):
        """
        Resolve o modelo usando Gurobi.
        
        Args:
            time_limit: Tempo limite em segundos (padrão: 600s = 10min)
            threads: Número de threads a usar
            verbose: Se True, mostra output do Gurobi
        
        Returns:
            Dict com os resultados da solução
        """
        print("\n" + "="*60)
        print("RESOLVENDO COM GUROBI (Branch & Bound)")
        print("="*60)
        
        # Configurar parâmetros
        self.model.Params.TimeLimit = time_limit
        self.model.Params.Threads = threads
        if not verbose:
            self.model.Params.OutputFlag = 0
        
        # Resolver
        start_time = time.time()
        self.model.optimize()
        self.solve_time = time.time() - start_time
        
        # Processar resultados
        if self.model.Status == GRB.OPTIMAL:
            print(f"\n✓ SOLUÇÃO ÓTIMA ENCONTRADA!")
            self.objective_value = self.model.ObjVal
            self._extract_solution()
            return self._get_results()
        
        elif self.model.Status == GRB.TIME_LIMIT:
            print(f"\n⚠ LIMITE DE TEMPO ATINGIDO")
            if self.model.SolCount > 0:
                print(f"  Melhor solução encontrada (não necessariamente ótima)")
                self.objective_value = self.model.ObjVal
                self._extract_solution()
                return self._get_results()
            else:
                print(f"  Nenhuma solução viável encontrada")
                return None
        
        else:
            print(f"\n✗ ERRO: Status = {self.model.Status}")
            return None
    
    def _extract_solution(self):
        """Extrai a solução das variáveis."""
        self.solution = {}
        
        for (patient_id, ward_name, d), var in self.y.items():
            if var.X > 0.5:  # Variável binária = 1
                self.solution[patient_id] = {
                    'ward': ward_name,
                    'day': d,
                    'patient_data': self.data.patients[patient_id]
                }
    
    def _get_results(self):
        """Retorna um dicionário com os resultados."""
        return {
            'objective_value': self.objective_value,
            'solve_time': self.solve_time,
            'solution': self.solution,
            'num_patients': len(self.solution),
            'gap': self.model.MIPGap if hasattr(self.model, 'MIPGap') else None
        }
    
    def print_solution(self, max_patients=10):
        """Imprime a solução de forma legível."""
        if not self.solution:
            print("Nenhuma solução disponível.")
            return
        
        print("\n" + "="*60)
        print("SOLUÇÃO")
        print("="*60)
        print(f"Valor objetivo: {self.objective_value:.2f}")
        print(f"Tempo de resolução: {self.solve_time:.2f}s")
        print(f"Pacientes alocados: {len(self.solution)}")
        
        print(f"\nPrimeiros {max_patients} pacientes:")
        print("-" * 60)
        
        for i, (patient_id, alloc) in enumerate(list(self.solution.items())[:max_patients]):
            patient = alloc['patient_data']
            print(f"\n{patient_id}:")
            print(f"  Especialização: {patient['specialization']}")
            print(f"  Enfermaria: {alloc['ward']}")
            print(f"  Dia de admissão: {alloc['day']}")
            print(f"  Janela permitida: [{patient['earliest']}, {patient['latest']}]")
            print(f"  Permanência: {patient['los']} dias")


# Teste do modelo
if __name__ == "__main__":
    # Carregar dados
    print("Carregando dados...")
    data = PatientAllocationData('data/s0m0.dat')


    # Criar e resolver modelo
    model = PatientAllocationMILP(data, lambda1=0.5, lambda2=0.5)
    model.build_model()
    
    results = model.solve(time_limit=300, threads=4)
    
    if results:
        model.print_solution()