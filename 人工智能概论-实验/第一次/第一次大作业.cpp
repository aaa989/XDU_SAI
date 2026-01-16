#include <stdio.h>
void DFS(int,int,int);
int edge[50][50],visited[50][50],min=999;
//定义一个数组edge用于储存迷宫地图，地图中0代表可达，1代表不可达
//定义一个数组visited用来标记已经遍历过的点，min代表走出迷宫需要的最少步数 
int n,m,startx,starty,endx,endy;//定义出口和入口的横纵坐标 
int next[4][2]={ {0,1},{1,0},{0,-1},{-1,0} };//定义一个数组next用来表示下一步的方向 
int main()
{
	int i,j;//定义i代表迷宫地图每行的点数，j代表每列的点数 
     printf("请输入矩阵行数及列数：\n");
	 scanf ("%d%d",&n,&m);//输入 n行 m列 的迷宫地图 
     printf("请输入邻接矩阵：\n");
	for (i=1;i<=n;i++)
	{
		for (j=1;j<=m;j++)
			scanf ("%d",&edge[i][j]);
	}//输入迷宫地图，数组的下标从1开始
	printf("请输入入口及出口位置坐标：\n") ;
	scanf ("%d%d%d%d",&startx,&starty,&endx,&endy);//输入入口和出口的横纵坐标 
	DFS(startx,starty,0);//深度优先搜索函数 
	printf ("min=%d",min);
	return 0;
}

void DFS(int x,int y,int count)//传入当前位置坐标和已经走过的步数 
{
	if (x==endx&&y==endy) 
	{
		if (count<min)//如果已经到达出口，判断该路径是否为当前最短路径 
			min=count;
		return;//递归结束，返回上一步
	} 
	int i,nextx,nexty;//下一步要走的位置坐标 
	for (i=0;i<4;i++)//利用循环来搜索所有可能的路径 
	{
		nextx=x+next[i][0];//计算下一步的位置坐标 
		nexty=y+next[i][1];
		if (nextx<1||nextx>n||nexty<1||nexty>m)//判断是否越界，未越界则继续执行 
			continue;
		if (edge[nextx][nexty]!=1&&visited[nextx][nexty]!=1) 
		{
			visited[nextx][nexty]=1;//如果下一步没越界，继续判断该位置是否为障碍物和是否已经遍历
			//若满足条件则标记已经遍历 
			DFS(nextx,nexty,count+1);//递归查找下一步骤，并将步数加一 
			visited[nextx][nexty]=0;//全部遍历过后，将路径的标记全部取消
			//走到出口执行该语句 
			//从出口进行回溯，最终回到入口 
		}
	}
}

