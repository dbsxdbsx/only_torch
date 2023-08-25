use only_torch::tensor::Tensor;

#[test]
fn test_ada_line() {
    let male_heights = Tensor::new_normal(171.0, 6.0, &[500]);
    let female_heights = Tensor::new_normal(158.0, 5.0, &[500]);

    let male_weights = Tensor::new_normal(70.0, 10.0, &[500]);
    let female_weights = Tensor::new_normal(57.0, 8.0, &[500]);

    let male_bfrs = Tensor::new_normal(16.0, 2.0, &[500]);
    let female_bfrs = Tensor::new_normal(22.0, 2.0, &[500]);

    let male_labels = Tensor::new(&vec![1.0; 500], &[500]);
    let female_labels = Tensor::new(&vec![-1.0; 500], &[500]);

    let mut train_set = Tensor::stack(
        &[
            &Tensor::stack(&[&male_heights, &female_heights], false),
            &Tensor::stack(&[&male_weights, &female_weights], false),
            &Tensor::stack(&[&male_bfrs, &female_bfrs], false),
            &Tensor::stack(&[&male_labels, &female_labels], false),
        ],
        true,
    );
    train_set.permute_mut(&[1, 0]);
    train_set.shuffle_mut(Some(0));

    for i in 0..5 {
        println!("{}", train_set.get(&[i]));
    }
}
