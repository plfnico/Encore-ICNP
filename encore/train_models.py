import os
import sys
import argparse
import datetime

if __name__ == '__main__':
    os.chdir('/mnt/ssd1/encore/open-source')#TODO: change to ur own dir
    if '/mnt/ssd1/encore/open-source' not in sys.path: sys.path.insert(0, '/mnt/ssd1/encore/open-source')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, dest='model', default='all')
    args = parser.parse_args()
    models = [args.model] if args.model != 'all' else ['cvae', 'gru']

    size_dir = './data/size/'
    interval_dir = './data/interval/'
    files = os.listdir(size_dir)
    apps = [file.strip('.txt') for file in files]

    date = datetime.datetime.now()
    for model in models:
        model_dir = '{model}-{year}-{month}-{day}-{hour}'.format(model=model, year=date.year, month=date.month, day=date.day, hour=date.hour)
        if os.path.exists('checkpoints/{dir}/'.format(dir=model_dir)):
            os.system('rm -r checkpoints/{dir}/'.format(dir=model_dir))
        os.makedirs('checkpoints/{dir}/'.format(dir=model_dir))
        print('start training Encore-{} in {}-{}-{} {}:{}:{:02d}'.format(model, date.year, date.month, date.day, date.hour, date.minute, date.second))
        sys.stdout.flush()
        for app in apps:
            command = 'python encore/train_{}.py --app {} --dir {}'.format(model, app, model_dir)
            os.system(command)