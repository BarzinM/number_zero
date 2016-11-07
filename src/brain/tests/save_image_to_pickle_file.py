from brain.prepros import ls, shuffleFiles, showMultipleArraysHorizontally, fileListToArray,arrayToFile,scaleData, dataGenerator


path = '/home/barzinm/Pets/deeplearning/assignments/notMNIST_small/'
files = [ls(path + s, '*.png')
         for s in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']]
files = sum(files, [])
files = shuffleFiles(files)


big_batch_size = 512

array = fileListToArray(files[:15000])
array = scaleData(array, 255)
train_dataset = '../data/processed/notmnist_images_train'
arrayToFile(train_dataset,array,big_batch_size)

array = fileListToArray(files[15000:18000])
array = scaleData(array, 255)
valid_dataset = '../data/processed/notmnist_images_valid'
arrayToFile(valid_dataset,array,big_batch_size)

gen = dataGenerator(8,train_dataset)
array = next(gen)+.5
showMultipleArraysHorizontally(array, max_per_row=4)