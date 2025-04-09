CC = mpicc
CFLAGS = -g -Wall
LDFLAGS = -lm

# 基本对象文件
POISSOBJS = decomp1d.o jacobi.o

# 辅助函数对象文件 (来自main.c的函数实现)
AUXOBJS = auxfuncs.o

# 第三题新增的对象文件
Q3OBJS = gather.o write_grid.o

# 第四题的对象文件
Q4OBJS = jacobi2d.o $(POISSOBJS) $(AUXOBJS) $(Q3OBJS)

EXECS = poiss1d poiss1dnb poissq2 poissq3 poissq4

all: $(EXECS)

# 第一题
poiss1d: main_q1.o $(POISSOBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# 第一题的非阻塞版本
poiss1dnb: mainnb.o $(POISSOBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# 第二题，链接辅助函数
poissq2: main_q2.o $(POISSOBJS) $(AUXOBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# 第三题，链接所有必要的对象文件
poissq3: main_q3.o $(POISSOBJS) $(AUXOBJS) $(Q3OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# 第四题目标
poissq4: main_q4.o $(Q4OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# 从main.c中提取辅助函数
auxfuncs.o: auxfuncs.c
	$(CC) $(CFLAGS) -c $< -o $@

# 第三题新增的文件编译
gather.o: gather.c gather.h decomp1d.h
	$(CC) $(CFLAGS) -c $< -o $@

write_grid.o: write_grid.c write_grid.h
	$(CC) $(CFLAGS) -c $< -o $@

# 编译第四题所需的文件
jacobi2d.o: jacobi2d.c jacobi2d.h
	$(CC) $(CFLAGS) -c $< -o $@
.PHONY: clean run_q3 plot

clean:
	rm -f *.o $(EXECS) *.txt poisson_sol_nx* poisson_*_sol_* poisson2d_sol_nx*_np* *.png

# 运行第三题的便捷命令
run_q3:
	mpirun -np 4 ./poissq3

# 运行Python绘图脚本
plot:
	python plot_solution.py

# 运行并生成热图
run_q3_plot: run_q3 plot

# 运行第四题的便捷命令（4个进程）
run_q4_4:
	mpirun -np 4 ./poissq4

# 运行第四题的便捷命令（16个进程）
run_q4_16:
	mpirun -np 16 ./poissq4

# 不同配置的测试运行
# 参数1: 格点大小 (默认31)
# 参数2: 是否使用非阻塞通信 (0:阻塞，1:非阻塞，默认1)
# 参数3: 是否使用重叠计算 (0:不重叠，1:重叠，默认1)
test_q4_configs:
	@echo "Testing with 4 processes, blocking communication, no overlap"
	mpirun -np 4 ./poissq4 31 0 0
	@echo "Testing with 4 processes, non-blocking communication, no overlap"
	mpirun -np 4 ./poissq4 31 1 0
	@echo "Testing with 4 processes, non-blocking communication, with overlap"
	mpirun -np 4 ./poissq4 31 1 1
	@echo "Testing with 16 processes, non-blocking communication, with overlap"
	mpirun -np 16 ./poissq4 31 1 1

# 运行Python绘图脚本
plot_q4:
	python plot_q4.py

# 依次运行Q4和绘图
run_q4_plot: run_q4_4 run_q4_16 plot_q4
