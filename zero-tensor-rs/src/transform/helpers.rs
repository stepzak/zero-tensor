pub fn is_float_int(val: f64) -> bool {
    val.fract() == 0.0
}
