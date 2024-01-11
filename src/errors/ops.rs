use std::fmt::{self, Display};

/// 张量的二元运算符
#[derive(Debug, PartialEq, Eq)]
pub enum Operator {
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Mul,
    MulAssign,
    Div,
    DivAssign,
    DotSum,
}
impl Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let operation_name = match self {
            Operator::Add => "相加",
            Operator::AddAssign => "自相加",
            Operator::Sub => "相减",
            Operator::SubAssign => "自相减",
            Operator::Mul => "相乘",
            Operator::MulAssign => "自相乘",
            Operator::Div => "相除",
            Operator::DivAssign => "自相除",
            Operator::DotSum => "点积和",
        };
        write!(f, "{}", operation_name)
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
            ComparisonOperator::GreaterOrEqual => "≥",
            ComparisonOperator::LessOrEqual => "≤",
            ComparisonOperator::GreaterThan => ">",
            ComparisonOperator::LessThan => "<",
            ComparisonOperator::Equal => "==",
            ComparisonOperator::NotEqual => "!=",
        };
        write!(f, "{}", operator_name)
    }
}
