import cv2
from pathlib import Path


DEBUG = True
rank = {}


def glob(path: Path, pattern: str='**/*.jpg'):
    return path.glob(pattern)


def save(img, output_path: Path, caption: str='Image.jpg'):
    cv2.imwrite(str(output_path / caption), img)


def update_rank(item: dict):
    assert len(item) == 1, 'item长度应为1'
    if len(rank) <= 1000:
        rank.update(item)
    else:
        for k in item:
            if k > min(rank):
                del rank[min(rank)]
                rank.update(item)


def detection(input_path: Path, output_path: Path):
    for path in glob(input_path, '**/*.jpg'):
        img = cv2.imread(str(path))
        try:
            img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片压缩为单通道的灰度图
        except cv2.error:
            continue
        img_resize = cv2.resize(img2gray, (600, 600))  # 为方便与其他图片比较可以将图片resize到同一个大小
        score = cv2.Laplacian(img_resize, cv2.CV_64F).var()
        print("Laplacian score of given image is ", score)
        if score > 100:  # 这个值可以根据需要自己调节，在我的测试集中100可以满足我的需求。
            print("%s: Good image!" % path) if DEBUG else None
            update_rank({str(int(score)): path})
        else:
            print("%s: Bad image!" % path) if DEBUG else None
            save(img, output_path / 'Bad_Images', path.name)
            # input('ok?: ')
    if len(rank):

        for p in rank.values():
            img = cv2.imread(str(p))
            save(img, output_path / 'Good_Images', p.name)
    print('All Done !')


if __name__ == '__main__':
    input_path = Path('../../../data/input')
    output_path = Path('../../../data/output')
    detection(input_path, output_path)
