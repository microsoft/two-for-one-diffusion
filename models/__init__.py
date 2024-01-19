from models.graph_transformer import GraphTransformer


def get_model(args, trainset, device):
    if args.backbone_network == "graph-transformer":
        model = GraphTransformer(
            trainset.num_beads,
            hidden_nf=args.hidden_features_gnn,
            device=device,
            n_layers=args.num_layers_gnn,
            use_intrinsic_coords=args.use_intrinsic_coords,
            use_abs_coords=args.use_abs_coords,
            use_distances=args.use_distances,
            conservative=args.conservative,
        )
    else:
        raise Exception(f"Network { args.backbone_network} not implemented")
    return model
