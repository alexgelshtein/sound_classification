import turicreate as tc
import os

# for dir in os.listdir('data/'):
#     print(f'{dir} - ', len(os.listdir(f'data/{dir}')))

data = tc.SFrame()
dir_count = 0
for dir in os.listdir('data/'):
    count = 0
    new_data = tc.load_audio('data/{}/'.format(dir))
    new_data['filename'] = new_data['path'].apply(lambda x: os.path.basename(x))
    new_data['category'] = '{}'.format(dir)
    new_column = [2 for _ in range(4)] + [1 for _ in range(len(new_data) - 4)]
    new_data = new_data.add_column(new_column, 'set')
    data = data.append(new_data)
    dir_count += 1
    print('[+] Processed dir {}'.format(dir_count))

test_set = data.filter_by(2, 'set')
train_set = data.filter_by(1, 'set')
print('[INFO] Test and train sets ready')

model = tc.sound_classifier.create(train_set, target='category', feature='audio', validation_set=None, max_iterations=100)
print('[INFO] Created model')

predictions = model.predict(test_set)
print('[INFO] Generated predictions')

metrics = model.evaluate(test_set)
print(metrics)

model.save('audio.model')