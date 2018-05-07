from PIL import Image
import numpy as np
import os

def miou(predict, ground_truth):
	# ground_truth = Image.open(ground_truth)
	# predict = Image.open(predict).resize((ground_truth.shape[0], ground_truth.shape[1]))
	
	g = np.array(Image.open(ground_truth))[:, :, 0]
	# print(g.shape)
	p = np.array(Image.open(predict).resize((g.shape[0],g.shape[0])))[:, :, 0]
	# print(p.shape)
	# exit(0)
	# print(p>=254)
	p[p< 250] = 0
	p[p>=250] = 1
	u_cnt = 0
	i_cnt = 0
	g = g[:,400:400+g.shape[0]]
	# Image.fromarray(g).show()
	# Image.fromarray(p).show()
	print(g.shape)
	print(p.shape)
	# print(g)
	# exit(0)
	for gg,pp in zip(g.ravel(), p.ravel()):
		if gg == 1 and pp == 1:
			i_cnt += 1
		if gg == 1 or pp == 1:
			u_cnt += 1
	print(i_cnt/u_cnt)
	return i_cnt/u_cnt
	# print(predict)

if __name__ == '__main__':
	g_folder = 'new_sheep_val_seg/'
	p_folder = 'predict_target/'
	p_files = os.listdir(p_folder)
	g_files = os.listdir(g_folder)
	p_files.sort()
	g_files.sort()
	# print(p_files, g_files)
	# exit(0)
	a = []
	for i,j in zip(p_files, g_files):
		print(i, j)
		x = miou(p_folder+i,g_folder+j)
		a.append(x)
	print(min(a))
	print(max(a))
	print(sum(a)/len(a))
