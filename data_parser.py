"""
Parser para ler os ficheiros .dat do problema de alocação de pacientes.
Este módulo processa os dados e organiza-os em estruturas Python facilmente utilizáveis.
"""

class PatientAllocationData:
    """Classe para armazenar todos os dados do problema."""
    
    def __init__(self, filepath):
        """
        Inicializa e carrega os dados do ficheiro .dat.
        
        Args:
            filepath: Caminho para o ficheiro .dat
        """
        self.filepath = filepath
        
        # Parâmetros gerais
        self.seed = None
        self.M = None  # Número de especializações menores por enfermaria
        self.weight_overtime = None
        self.weight_undertime = None
        self.weight_delay = None
        self.num_days = None
        
        # Especializações
        self.specialisms = {}  # {nome: {'workload_factor': float, 'ot_time': list}}
        
        # Enfermarias
        self.wards = {}  # {nome: {capacidade, workload_cap, major_spec, minor_specs, carryover_patients, carryover_workload}}
        
        # Pacientes
        self.patients = {}  # {nome: {specialization, earliest, latest, los, surgery_duration, workload_per_day}}
        
        # Carregar dados
        self._parse_file()
    
    def _parse_file(self):
        """Lê o ficheiro e extrai todos os dados."""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # PARSE GENERAL INFO
            if line.startswith('Seed:'):
                self.seed = int(line.split(':')[1].strip())
            
            elif line.startswith('M:'):
                self.M = int(line.split(':')[1].strip())
            
            elif line.startswith('Weight_overtime:'):
                self.weight_overtime = float(line.split(':')[1].strip())
            
            elif line.startswith('Weight_undertime:'):
                self.weight_undertime = float(line.split(':')[1].strip())
            
            elif line.startswith('Weight_delay:'):
                self.weight_delay = float(line.split(':')[1].strip())
            
            elif line.startswith('Days:'):
                self.num_days = int(line.split(':')[1].strip())
            
            # PARSE FOR SPECIALISMS
            elif line.startswith('Specialisms:'):
                num_specialisms = int(line.split(':')[1].strip())
                # Próximas linhas contêm as especializações
                for j in range(1, num_specialisms + 1):
                    spec_line = lines[i + j].strip().split('\t')
                    spec_name = spec_line[0]
                    workload_factor = float(spec_line[1])
                    ot_time = [int(x) for x in spec_line[2].split(';')]
                    
                    self.specialisms[spec_name] = {
                        'workload_factor': workload_factor,
                        'ot_time': ot_time
                    }
                i += num_specialisms
            
            # PARSE FOR WARDS
            elif line.startswith('Wards:'):
                num_wards = int(line.split(':')[1].strip())
                # Próximas linhas contêm as enfermarias
                for j in range(1, num_wards + 1):
                    ward_line = lines[i + j].strip().split('\t')
                    ward_name = ward_line[0]
                    bed_capacity = int(ward_line[1])
                    workload_capacity = float(ward_line[2])
                    major_spec = ward_line[3]
                    minor_specs = ward_line[4].split(';') if ward_line[4] != 'NONE' else []
                    carryover_patients = [int(x) for x in ward_line[5].split(';')]
                    carryover_workload = [float(x) for x in ward_line[6].split(';')]
                    
                    self.wards[ward_name] = {
                        'bed_capacity': bed_capacity,
                        'workload_capacity': workload_capacity,
                        'major_specialization': major_spec,
                        'minor_specializations': minor_specs,
                        'carryover_patients': carryover_patients,
                        'carryover_workload': carryover_workload
                    }
                i += num_wards
            
            # PARSE FOR PATIENTS
            elif line.startswith('Patients:'):
                num_patients = int(line.split(':')[1].strip())
                # Próximas linhas contêm os pacientes
                for j in range(1, num_patients + 1):
                    if i + j >= len(lines):
                        break
                    patient_line = lines[i + j].strip().split('\t')
                    if len(patient_line) < 7:
                        continue
                        
                    patient_id = patient_line[0]
                    specialization = patient_line[1]
                    earliest = int(patient_line[2])
                    latest = int(patient_line[3])
                    los = int(patient_line[4])  # Length of stay
                    surgery_duration = int(patient_line[5])
                    workload = [float(x) for x in patient_line[6].split(';')]
                    
                    self.patients[patient_id] = {
                        'specialization': specialization,
                        'earliest': earliest,
                        'latest': latest,
                        'los': los,
                        'surgery_duration': surgery_duration,
                        'workload_per_day': workload
                    }
                i += num_patients
            
            i += 1
    
    def print_summary(self):
        """Imprime um resumo dos dados carregados."""
        print("=" * 60)
        print("RESUMO DOS DADOS")
        print("=" * 60)

        # GENERAL INFO
        print(f"Período de planeamento: {self.num_days} dias")
        print(f"Especializações menores por enfermaria (M): {self.M}")
        print(f"Pesos: Overtime={self.weight_overtime}, Undertime={self.weight_undertime}, Delay={self.weight_delay}")
        print(f"\nNúmero de especializações: {len(self.specialisms)}")
        print(f"Número de enfermarias: {len(self.wards)}")
        print(f"Número de pacientes: {len(self.patients)}")
        
        # SPECIALISM INFO
        print("\n" + "-" * 60)
        print("Especializações:")
        print("-" * 60)
        for spec_name, spec_data in self.specialisms.items():
            print(f"\n{spec_name}:")
            print(f"  Fator de carga de trabalho: {spec_data['workload_factor']}")
            for i in range(0, self.num_days):
                print(f"  OT (dia {i}) : {spec_data['ot_time'][i]}")
                print("-" * 30)

        # WARDS INFO
        print("\n" + "-" * 60)
        print("ENFERMARIAS:")
        print("-" * 60)
        for ward_name, ward_data in self.wards.items():
            print(f"\n{ward_name}:")
            print(f"  Capacidade de camas: {ward_data['bed_capacity']}")
            print(f"  Capacity de trabalho: {ward_data['workload_capacity']}")
            print(f"  Especialização principal: {ward_data['major_specialization']}")
            print(f"  Especializações menores: {ward_data['minor_specializations']}")
            for i in range(0, self.num_days):
                print(f"  Pacientes pré-existentes (dia {i}) : {ward_data['carryover_patients'][i]}")
                print(f"  Trabalho pré-existente (dia {i}) : {ward_data['carryover_workload'][i]}")
                print("-" * 30)
        
        # PATIENTS INFO
        print("\n" + "-" * 60)
        print("AMOSTRA DE PACIENTES (primeiros 5):")
        print("-" * 60)
        for i, (patient_id, patient_data) in enumerate(list(self.patients.items())[:5]):
            print(f"\n{patient_id}:")
            print(f"  Especialização: {patient_data['specialization']}")
            print(f"  Janela de admissão: [{patient_data['earliest']}, {patient_data['latest']}]")
            print(f"  Duração do internamento: {patient_data['los']} dias")
            print(f"  Duração da cirurgia: {patient_data['surgery_duration']} minutos")
            for i in range(0, patient_data['los']):
                print(f"  Carga de trabalho (dia {i}): {patient_data['workload_per_day'][i]}")
        
        print("\n" + "=" * 60)


# Teste do parser
if __name__ == "__main__":
    # Carregar os dados
    data = PatientAllocationData('data/s0m0.dat'')
    
    # Imprimir resumo
    data.print_summary()
