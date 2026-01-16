#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 选择排序
void selectionSort(int a[], int n) {
	int i, j, mi, t;//mi最小数t表示一个暂存数字
	for (i = 0; i < n - 1; i++) {
		mi = i;
		for (j = 0; j < n - i - 1; j++) {
			if (a[j] < a[mi])
				mi = j;
		}
		t = a[mi];//换位置
		a[mi] = a[i];
		a[i] = t;
	}
}

// 快速排序
void quickSort(int a[], int l, int h) {
	if (l < h) {
		int p = a[h];
		int i = (l - 1);
		for (int j = l; j < h; j++) {
			if (a[j] < p) {
				i++;
				int t = a[i];
				a[i] = a[j];
				a[j] = t;
			}
		}
		int t = a[i + 1];//大的一组换位置
		a[i + 1] = a[h];
		a[h] = t;
		int partitionIndex = i + 1;
		quickSort(a, l, partitionIndex - 1);
		quickSort(a, partitionIndex + 1, h);
	}
}

// 记录排序耗时
double Timexz( int a[], int n) {//选择排序时间
	clock_t start, end;
	double timeSpent;
	start = clock();
	selectionSort(a, n);
	end = clock();
	timeSpent = (double)(end - start) / CLOCKS_PER_SEC;
	return timeSpent;
}

double Timeks( int a[], int l, int h) {//快速排序时间
	clock_t start, end;
	double timeSpent;
	start = clock();
	quickSort(a, l, h);
	end = clock();
	timeSpent = (double)(end - start) / CLOCKS_PER_SEC;
	return timeSpent;
}

int main() {
	int nS = 100;
	int nL = 100000;
	int smallA[nS];
	int largeA[nL];
	// 初始化随机序列
	srand(time(NULL));
	for (int i = 0; i < nS; i++) {
		smallA[i] = rand() % 100;//生成小数据
	}
	for (int i = 0; i < nL; i++) {
		largeA[i] = rand() % 1000000;//生成大数据
	}
	// 测量并打印耗时
	double xsjxzpx = Timexz(smallA, nS);//小数据选择排序
	double dsjxzpx = Timexz(largeA, nL);//大数据选择排序
	double xsjkspx = Timeks(smallA, 0, nS - 1); //小数据快速排序
	double dsjkspx = Timeks(largeA, 0, nL - 1); //大数据快速排序
	printf("选择排序 - 小数据量耗时: %f 秒\n", xsjxzpx);
	printf("选择排序 - 大数据量耗时: %f 秒\n", dsjxzpx);
	printf("快速排序 - 小数据量耗时: %f 秒\n", xsjkspx);
	printf("快速排序 - 大数据量耗时: %f 秒\n", dsjkspx);
	return 0;
}
