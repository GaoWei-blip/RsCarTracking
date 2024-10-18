import os
import motmetrics as mm


def computer_score(ground_path, pred_path):
    mh = mm.metrics.create()
    accs = []
    names = []

    for filename in os.listdir(ground_path):
        if filename.endswith(".csv"):
            file_root, _ = os.path.splitext(filename)
            file_root = file_root[:-3]

            gt = mm.io.loadtxt(ground_path + f'{file_root}-gt.csv', fmt="mot15-2D")
            ts = mm.io.loadtxt(pred_path + f'{file_root}.txt', fmt="mot15-2D")

            accs.append(mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5))
            names.append(filename)

    metrics = list(mm.metrics.motchallenge_metrics)
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    # print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

    idf1 = summary.loc[:, 'idf1'].iloc[-1]
    mota = summary.loc[:, 'mota'].iloc[-1]
    score = (idf1 + mota) / 2 * 100
    print(f"idf1: {idf1}")
    print(f"mota: {mota}")
    print(f"Score: {score}")

if __name__ == '__main__':

    computer_score('./data/train1/', './output/results/')
