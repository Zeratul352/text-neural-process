import pyconll
import csv


full_train = pyconll.load_from_file('./datasets/ru_syntagrus-ud-train.conllu')

csv_name = './datasets/manifest.csv'
split_number = 5

"""
with open(csv_name, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['number', 'filename'])
"""
header = ['number', 'filename']
manifest_data = []
numfile = 0
part_name = "./datasets/train_part_" + str(numfile) + '.conllu'
part_sentence = pyconll.load_from_file('./datasets/ru_syntagrus-ud-train.conllu')
part_sentence.clear()
internal_index = 0
for token in full_train:
    part_sentence.append(token)
    internal_index = internal_index + 1
    if internal_index == len(full_train)//split_number:
        internal_index = 0
        with open(part_name, 'w', encoding='utf-8', newline='') as f:
            part_sentence.write(f)

        manifest_data.append([numfile, part_name])
        """
        with open(csv_name, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([str(numfile), part_name])
        """
        part_sentence.clear()
        numfile = numfile + 1
        part_name = "./datasets/train_part_" + str(numfile) + '.conllu'


with open(csv_name, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(manifest_data)
"""
if internal_index != 0:
    internal_index = 0
    with open(part_name, 'w', encoding='utf-8', newline='') as f:
        part_sentence.write(f)

    with open(csv_name, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([str(numfile), part_name])

    part_sentence.clear()
    numfile = numfile + 1
    part_name = "./datasets/train_part_" + str(numfile) + '.conllu'
"""





