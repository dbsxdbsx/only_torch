pub trait FloatTrait {
    fn is_integer(&self) -> bool;
}

impl FloatTrait for f32 {
    fn is_integer(&self) -> bool {
        self.fract() < Self::EPSILON
    }
}
