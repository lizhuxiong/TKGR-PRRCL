#include <bits/stdc++.h>
#include <cmath>
#define N 100050
#define N2 20
using namespace std;

const double eps = 1e-5;

int num_of_walks = 40;
int seq_len = 6;
int o[20];
char dis[N][N];
bool vis[N];
int n, m;

vector<int> E[N];
vector<double> p_merw[N];
vector<int> T[N];
int timestamps[20];
float t[20];
int totalEpochs = 15;


class AliasTable{
    public:
        vector<int> A, B;
        vector<double> S;
    public:
        AliasTable () {}
        // init Alias Table。
        void init(vector<int> &a, vector<double> &p) {
            queue<int> qA, qB;
            queue<double> pA, pB;
            int n = (int)a.size();

            for (int i=0;i<n;i++) p[i] = p[i] * n;
            for (int i=0;i<n;i++)
                if (p[i] > 1.0) {
                    qA.push(a[i]);
                    pA.push(p[i]);
                } else {
                    qB.push(a[i]);
                    pB.push(p[i]);
                }
            while (!qA.empty() && !qB.empty()) {
                int idA = qA.front(); qA.pop();
                double probA = pA.front(); pA.pop();
                int idB = qB.front(); qB.pop();
                double probB = pB.front(); pB.pop();

                A.push_back(idA);
                B.push_back(idB);
                S.push_back(probB);

                double res = probA-(1.0-probB);

                if (abs(res-1.0) < eps) {
                    A.push_back(idA);
                    B.push_back(idA);
                    S.push_back(res);
                    continue;
                }
                if (res > 1.0) {
                    qA.push(idA);
                    pA.push(res);
                } else {
                    qB.push(idA);
                    pB.push(res);
                }
            }

            while (!qA.empty()) {
                int idA = qA.front(); qA.pop();
                pA.pop();
                A.push_back(idA);
                B.push_back(idA);
                S.push_back(1.0);
            }

            while (!qB.empty()) {
                int idB = qB.front(); qB.pop();
                pB.pop();
                A.push_back(idB);
                B.push_back(idB);
                S.push_back(1.0);
            }
        }
        int roll() {
            if ((int)A.size() == 0) {
		        return -1;
	        }
            int x = rand() % ((int)A.size());
            double p = 1.0 * rand() / RAND_MAX;
            return p>S[x] ? A[x] : B[x];
        }

}AT[N];

void link(int u, int v, int t, double p)
{
    E[u].push_back(v);
    T[u].push_back(t); //add time
    p_merw[u].push_back(p);
}


void bfs(int S)
{
    queue<int> q;
    q.push(S);
    dis[S][S] = 1;
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        if (dis[S][u] > seq_len)
            return;
        for (int i = 0; i < (int)E[u].size(); i++)
        {
            int v = E[u][i];
            if (dis[S][v] == 0)
            {
                dis[S][v] = dis[S][u] + 1;
                q.push(v);
            }
        }
    }
    return;
}

int findIndex(const std::vector<int> subvector, int value) {
    for (size_t i = 0; i < subvector.size(); ++i) {
        if (subvector[i] == value) {
            return i;
        }
    }
    return -1;
}

int main(int argc, char *argv[])
{

    if (argc != 4)
    {
        cerr << "ERROR: Incorrect number of parameters. " << endl;
        return 0;
    }

    stringstream ss1, ss2;
    ss1.str("");
    ss1 << "../edge_input/";
    ss1 << argv[1];
    ss1 << "/";
    ss1 << argv[1];
//    ss1 << "_nsl";
    ss1 << ".in";

    // freopen(ss1.str().c_str(), "r", stdin);
    FILE* newStdin = freopen(ss1.str().c_str(), "r", stdin);

    if (newStdin == NULL) {
        std::cerr << "Error: Failed to reopen stdin with file " << ss1.str() << std::endl;
        return 1;
    }

    num_of_walks = atoi(argv[2]);
    seq_len = atoi(argv[3]);

    ss2.str("");
    ss2 << "../path_data/";
    // ss2 << "/data/syf/rw/";
    ss2 << argv[1];
    ss2 << "/";
    ss2 << argv[1];
    ss2 << "_";
    ss2 << argv[2];
    ss2 << "_";
    ss2 << argv[3];
//    ss2 << "_";
//    ss2 << "nsl";
    ss2 << "_merw.txt";

    cout << "File input: " << ss1.str().c_str() << endl;
    cout << "File output: " << ss2.str().c_str() << endl;

    freopen(ss2.str().c_str(), "w", stdout);
    srand(time(0));
    scanf("%d%d", &n, &m);

    cerr << argv[1] << ": " << n << endl;

    cerr << "total tuples: " << m << endl;

    for (int i = 1; i <= m; i++)
    {
        int u, v, t;
        double p;
        scanf("%d%d%d%lf", &u, &v, &t, &p);
        link(u, v, t, p);
    }

    for (int i=0;i<n;i++) {
        AT[i].init(E[i], p_merw[i]);
    }

    for (int i = 0; i < n; i++)
        bfs(i);

    // int totalEpochs = 1000;
    int totalEpochs = 1;
    int progress = 0;
    for (int epoch = 0; epoch < totalEpochs; epoch++)
    {
        for (int st = 0; st < n; st++)
        {
            for (int i = 0; i < num_of_walks; i++)
            {
                int u = st;
                int prior = st;
                int index = 0;
                int cur_time = -1;
                printf("[");
                for (int _ = 0; _ < seq_len; _++)
                {
                    printf("%d", u);
//                    o[_] = dis[st][u];
                    if (_ != 0){
                        index = findIndex(E[prior], u);
                        if (index == -1){
                            index = findIndex(E[u], prior);
                        }
                        if (index == -1){
                             timestamps[0] = -1;
                             t[_] = 1; //
                             timestamps[_] = -1;  // no path -> exapmle: [7120, 7120, 7120, -1, -1 , -1, 1, 1, 1]
                        }else{
                            if(_ == 1){
                                cur_time = T[prior][index];
                                timestamps[0] = cur_time;
                            }
                            t[_] = exp(-0.1 * abs(cur_time - T[prior][index])) * p_merw[prior][index] + std::sin(T[prior][index]);
                            timestamps[_] = T[prior][index];
                        }

                    }else{
                        t[_] = 1;
                    }
                    printf(", ");
                    int g = AT[u].roll();
                    if(g == -1){
                        continue;
                    }
                    prior = u;
                    u = g;
                }

//                for (int _ = 0; _ < seq_len; _++)
//                {
//                    printf("%d", o[_] - 1);
//                    // if (_ != seq_len - 1)
//                    printf(", ");
//                }

                for (int _ = 0; _ < seq_len; _++)
                {
                    printf("%d", timestamps[_]);
                    printf(", ");
                }
                for (int _ = 0; _ < seq_len; _++)
                {
                    // printf("%d", o[_] - 1);
                    printf("%.4f", t[_]);
                    if (_ != seq_len - 1)
                        printf(", ");
                }

                printf("]\n");
            }
        }

        progress = static_cast<int>((static_cast<float>(epoch) / totalEpochs) * 100);

        cerr << "[";
        for (int i = 0; i < 50; i++) {
            if (i < progress / 2) {
                cerr << "=";
            } else {
                cerr << " ";
            }
        }
        cerr << "] " << progress << "%\r";

    }
    cerr << endl;
    fclose(stdin);
    fclose(stdout);
    return 0;
}