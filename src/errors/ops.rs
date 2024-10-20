use std::fmt::{self, Display};

/// 张量的二元运算符
#[derive(Debug, PartialEq, Eq)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    DotSum,
}
impl Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let operation_name = match self {
            Self::Add => "相加",
            Self::Sub => "相减",
            Self::Mul => "相乘",
            Self::Div => "相除",
            Self::DotSum => "点积和",
        };
        write!(f, "{operation_name}")
    }
}

/// 比较运算符
#[derive(Debug, PartialEq, Eq)]
pub enum ComparisonOperator {
    GreaterOrEqual,
    LessOrEqual,
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
}
impl Display for ComparisonOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let operator_name = match self {
            Self::GreaterOrEqual => "≥",
            Self::LessOrEqual => "≤",
            Self::GreaterThan => ">",
            Self::LessThan => "<",
            Self::Equal => "==",
            Self::NotEqual => "!=",
        };
        write!(f, "{operator_name}")
    }
}
