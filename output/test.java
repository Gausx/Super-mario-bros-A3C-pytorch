public clas DemoPrim {
    public static void main(String[] args) {
        //邻接矩阵
        graph = new int[][]{
                {0, 6, 1, 5, 0, 0},
                {6, 0, 5, 0, 3, 0},
                {1, 5, 0, 5, 6, 4},
                {5, 0, 5, 0, 0, 2},
                {0, 3, 6, 0, 0, 6},
                {0, 0, 4, 2, 6, 0}
                };
        int ans = prim(graph, 0);
        System.out.println(ans);
    }

    public static int prime(int[][] graph, int start) {
        boolean[] via = new boolean[];
    }
}