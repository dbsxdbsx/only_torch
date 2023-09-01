pub trait FloatTrait {
    fn is_integer(&self) -> bool;
}

impl FloatTrait for f32 {
    fn is_integer(&self) -> bool {
        self.fract() < f32::EPSILON
    }
}
