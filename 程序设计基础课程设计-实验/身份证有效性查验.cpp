#include <stdio.h>

int main() {
	char ID[19] = "\0";
	char M[11] = { '1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2' };
	int qz[17] = { 7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2 }; //权重
	int n = 0, sum = 0, flag = 1;//n可能输入的数据个数sum用来前十七位数求和运算flag用作一个布尔数，最终取0为有误，取1则全对
	printf("欢迎使用身份证有效性查验功能\n情况如下：\n");
	printf("如有问题，即输即得;\n如无误，则无任何输出，最终将提示全部正确。");
	printf("请输入查验个数：");
	scanf("%d", &n);
	getchar();
	for (int j = 0; j < n; j++) {
		printf("请输入身份证号：");
		gets(ID);

		for (int i = 0; i < 17; i++) {
			if (ID[i] > '9' || ID[i] < '0') {//由于身份证号除最后一位外都是0-9的数字构成，因此可进行初步判断；
				printf("有问题：");
				puts(ID);
				flag = 0;
				break;
			} else {
				sum = sum + ((ID[i] - '0') * qz[i]);//算加权数求和
			}

			if (i == 16) {
				if (ID[17] != M[sum % 11]) {//与校验码核对
					printf("有问题：");
					puts(ID);
					flag = 0;
				}
			}
		}

	}
	if (flag == 1) {
		printf("全部正确");
	}
}

