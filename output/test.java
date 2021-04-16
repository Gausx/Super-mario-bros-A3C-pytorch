class Solution {
    public int swimInWater(int[][] grid) {
        int n = grid.length;
        int m = grid[0].length;

        List<int[]> list = new ArrayList<>();

        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                int id = i * m + j;

                if(i < n - 1) {
                    list.add(new int[]{id, id + m, Math.max(grid[i][j], grid[i + 1][j])});
                }

                if(j < m - 1) {
                    list.add(new int[]{id, id + 1, Math.max(grid[i][j], grid[i][j + 1])});
                }
            }
        }
        Collections.sort(list, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return a[2] - b[2];
            }
        });

        Union union = new Union(n * m);

         for(int[] edge : list) {
            union.connect(edge[0], edge[1]);
            if(union.connected(0, n * m - 1)) {
                return edge[2];
            }
        }
        return 0;

    }

    class Union {
        private int count;
        private int[] parent;
        private int[] size;

        public Union(int n) {
            this.count = n;
            parent = new int[n];
            size = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                size[i] = 1;
            }
        }

        public void connect(int p, int q) {
            int rootP = find(p);
            int rootQ = find(q);
            if(rootP != rootQ) {
                if(size[rootP] > size[rootQ]) {
                    parent[rootQ] = rootP;
                    size[rootP] += size[rootQ];
                }
                else {
                    parent[rootP] = rootQ;
                    size[rootQ] += size[rootP];
                }
            }
            this.count--;
        }

        public boolean connected(int p, int q) {
            return find(p) == find(q);
        }

        public int find(int x) {
            while (x != parent[x]) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }



    }
}