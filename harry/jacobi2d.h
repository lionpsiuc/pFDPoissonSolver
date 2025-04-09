#define maxn 31 + 2

void exchang2d_1(double x[][maxn], int nx, int row_s, int row_e, int col_s,
                 int col_e, MPI_Comm comm, int nbrleft, int nbrright, int nbrup,
                 int nbrdown);

void sweep2d(double a[][maxn], double f[][maxn], int nx, int row_s, int row_e,
             int col_s, int col_e, double b[][maxn]);

double griddiff2d(double a[][maxn], double b[][maxn], int nx, int row_s,
                  int row_e, int col_s, int col_e);
