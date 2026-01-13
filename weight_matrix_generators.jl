# Function to generate the weight matrix of a hierarchical diamond lattice
function diamond_weight_matrix(level::Int)
    level ≥ 1 || error("level must be ≥ 1")

    # start with single bond
    edges = [(0, 1)]
    next_node = 2

    for _ in 1:(level - 1)
        new_edges = Tuple{Int,Int}[]
        for (u, v) in edges
            c = next_node
            d = next_node + 1
            next_node += 2

            # u--c--v and u--d--v
            push!(new_edges, (u, c))
            push!(new_edges, (c, v))
            push!(new_edges, (u, d))
            push!(new_edges, (d, v))
        end
        edges = new_edges
    end

    N = next_node
    W = zeros(Int, N, N)

    for (i, j) in edges
        W[i+1, j+1] = 1   # +1 for Julia indexing
        W[j+1, i+1] = 1
    end

    return W
end

function one_dimensional_line_weight_matrix(num_nodes::Int; ring::Bool=false)
    num_nodes ≥ 2 || error("num_nodes must be ≥ 2")

    W = zeros(Int, num_nodes, num_nodes)

    for i in 1:(num_nodes - 1)
        W[i, i+1] = 1
        W[i+1, i] = 1
    end

    if ring
        W[1, num_nodes] = 1
        W[num_nodes, 1] = 1
    end

    return W
end
