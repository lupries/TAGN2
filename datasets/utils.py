import yaml

def sequence_names(root='davis-2017/data/'):
  sequence_file = open(root+'db_info.yaml','r')
  sequences = yaml.load(sequence_file)['sequences']
  seq_names = [seq['name'] for seq in sequences if seq['year'] == 2016]
  return seq_names