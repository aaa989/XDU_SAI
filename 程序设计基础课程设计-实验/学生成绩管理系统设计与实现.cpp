#include <bits/stdc++.h>
using namespace std;

typedef struct {
	long long int id;
	char name[50];
	char gender[20];
	int score;
} Student;

Student students[100];
int num = 0;

void add() {
	cout << "请输入学生信息：" << endl;
	cout << "学号：" ;
	scanf("%d", &students[num].id);

	cout << "姓名：" ;
	scanf("%s", &students[num].name);
	cout << "性别：";
	scanf("%s", &students[num].gender);
	cout  << "C语言成绩：" ;
	scanf("%d", &students[num].score);

	num++;
	cout << "学生信息录入成功\n";

}

void find() {
	long long int m;
	int found = 0;
	cout << "请输出要查询的学号：";
	cin >> m;
	for (int i = 0; i < num; i++)
		if (students[i].id == m) {
			cout << "学号:" << students[i].id;
			cout << "  姓名:" << students[i].name;
			cout << "  性别:" << students[i].gender;
			cout << "  C语言成绩:" << students[i].score;
			found = 1;
		}
	if (found == 0)
		cout << "未找到该学号对应的学生\n";

}

void change() {
	int found = 0;
	long long int m;
	cout << "请输入要修改信息的学号：";
	cin >> m;
	for (int i = 0; i < num; i++)
		if (students[i].id == m) {
			found = 1;
			cout << endl << "请输出修改后的信息：" << endl;
			cout << "学号:";
			cin >> students[i].id;
			cout << "姓名:";
			cin >> students[i].name;
			cout << "性别:";
			cin >> students[i].gender;
			cout << "C语言成绩:";
			cin >> students[i].score;
		}
	if ( found == 0)
		cout << "未找到对应学号学生\n";
	else
		cout << "学生信息修改完成\n";
}

void detect() {
	long long int m;
	cout << "请输入要删除的学号： " << endl;
	cin >> m;
	int found = 0;
	for (int i = 0; i < num; i++)
		if (students[i].id == m) {
			found = 1;
			for (int j = i + 1; j <= num; j++)
				students[j - 1] = students[j];
		}
	if ( found == 0)
		cout << "未找到对应学号学生\n";
	else
		cout << "学生信息删除完成\n";

}

void Sort() {
	for (int i = 0; i < num - 1; i++) {
		for (int j = 0; j < num - i - 1 ; j++) {
			if (students[j].score < students[j + 1].score) {
				Student temp = students[j];
				students[j] = students[j + 1];
				students[j + 1] = temp;
			}
		}

	}
	cout << "学生成绩排序完成\n";

}

void cal() {
	int lo, up;
	int count = 0 ;
	cout << "请输出成绩统计的上下限：\n";
	cout << "下限:";
	cin >> lo;
	cout << "上限:";
	cin >> up;
	for (int i = 0; i < num; i++) {
		if (students[i].score >= lo &&  students[i].score <= up )
			count++;
	}
	cout << "成绩在" << lo << "和" << up << "之间的学生人数为：" << count << endl;
}

void put() {
	cout << "学生信息如下：\n";
	printf("学号\t姓名\t性别\tC语言程序\n");
	for (int i = 0; i < num; i++) {
		printf("%d\t%s\t%s\t%d\n", students[i].id, students[i].name, students[i].gender, students[i].score);
	}

}


int main() {
	cout << "------------------------------------" << endl;
	cout << "|    欢迎进入学生成绩管理系统      |" << endl;
	cout << "------------------------------------" << endl;
	cout << "|         1.学生信息录入           |" << endl;
	cout << "|         2.学生信息查询           |" << endl;
	cout << "|         3.学生信息修改           |" << endl;
	cout << "|         4.学生信息删除           |" << endl;
	cout << "|         5.学生信息排序           |" << endl;
	cout << "|         6.学生信息统计           |" << endl;
	cout << "|         7.学生信息输出           |" << endl;
	cout << "|         0.退出系统               |" << endl;
	cout << "------------------------------------" << endl;
	while (1) {
		cout << endl << "请选择功能0~7：";
		int n;
		cin >> n;
		if ( n == 1)
			add();
		else if ( n == 2)
			find();
		else if ( n == 3)
			change();
		else if ( n == 4)
			detect();
		else if (n == 5)
			Sort();
		else if (n == 6)
			cal();
		else if (n == 7)
			put();
		else
			return 0;
	}
	return 0;
}