private static MutableGraph<Integer> createGraph(EdgeType edgeType) {
    switch(edgeType) {
        case UNDIRECTED:
            return GraphBuilder.undirected().allowsSelfLoops(true).build();
        case DIRECTED:
            return GraphBuilder.directed().allowsSelfLoops(true).build();
        default:
            throw new IllegalStateException("Unexpected edge type: " + edgeType);
    }
}
