pub const VERSION: &str = "0.6.0";

struct HandshakeBuilder {
    cur_msg: String
}

impl HandshakeBuilder {
    pub fn new() -> Self {
        Self { cur_msg: format!("ZT {VERSION} ") }
    }

    pub fn add_key<T: std::fmt::Display>(mut self, key: &str, value: T) -> Self {
        let msg = format!("{key}={value} ");
        self.cur_msg += &msg;
        self
    }

    pub fn build(self) -> String {
        let mut s = self.cur_msg;
        s += "\n";
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_has_correct_prefix() {
        let builder = HandshakeBuilder::new();
        assert_eq!(builder.cur_msg, format!("ZT {VERSION} "));
    }

    #[test]
    fn test_build_empty_produces_just_prefix_with_newline() {
        let result = HandshakeBuilder::new().build();
        assert_eq!(result, format!("ZT {VERSION} \n"));
        assert!(result.ends_with('\n'));
    }

    #[test]
    fn test_add_single_key() {
        let result = HandshakeBuilder::new()
            .add_key("cb_size", 152)
            .build();
        
        assert!(result.starts_with("ZT 0.6.0 "));
        assert!(result.contains("cb_size=152"));
        assert!(result.ends_with('\n'));
    }

    #[test]
    fn test_add_multiple_keys_are_space_separated() {
        let result = HandshakeBuilder::new()
            .add_key("cb_size", 152)
            .add_key("head_offset", 0)
            .add_key("tail_offset", 64)
            .build();
        
        let parts: Vec<&str> = result.trim().split_whitespace().collect();
        assert_eq!(parts[0], "ZT");
        assert_eq!(parts[1], VERSION);
        assert_eq!(parts[2], "cb_size=152");
        assert_eq!(parts[3], "head_offset=0");
        assert_eq!(parts[4], "tail_offset=64");
    }

    #[test]
    fn test_chaining_api_returns_builder() {
        let _builder: HandshakeBuilder = HandshakeBuilder::new()
            .add_key("a", 1)
            .add_key("b", 2)
            .add_key("c", 3);
    }

    #[test]
    fn test_different_value_types() {
        let result = HandshakeBuilder::new()
            .add_key("int_val", 42usize)
            .add_key("str_val", "hello")
            .add_key("bool_val", true)
            .add_key("float_val", 3.14f64)
            .build();
        
        assert!(result.contains("int_val=42"));
        assert!(result.contains("str_val=hello"));
        assert!(result.contains("bool_val=true"));
        assert!(result.contains("float_val=3.14"));
    }

    #[test]
    fn test_build_appends_newline_exactly_once() {
        let result = HandshakeBuilder::new()
            .add_key("key", "value")
            .build();
        
        assert!(result.ends_with('\n'));
        assert_eq!(result.matches('\n').count(), 1);
    }

    #[test]
    fn test_build_is_consuming() {
        let builder = HandshakeBuilder::new().add_key("k", 1);
        let _ = builder.build();
    }

    #[test]
    fn test_real_world_handshake_format() {
        let result = HandshakeBuilder::new()
            .add_key("cb_size", 152)
            .add_key("head_offset", 0)
            .add_key("tail_offset", 64)
            .add_key("nslots", 4)
            .add_key("slot_size", 1024)
            .add_key("tensor_keys", "image,mask,label")
            .build();
        
        let parts: Vec<&str> = result.trim().split_whitespace().collect();
        assert_eq!(parts[0], "ZT");
        assert_eq!(parts[1], VERSION);
        
        for part in &parts[2..] {
            assert!(part.contains('='), "Part '{part}' is not key=value");
            let (key, value) = part.split_once('=').unwrap();
            assert!(!key.is_empty(), "Empty key in '{part}'");
            assert!(!value.is_empty(), "Empty value in '{part}'");
        }
    }

    #[test]
    fn test_consumer_can_parse_produced_handshake() {
        let handshake = HandshakeBuilder::new()
            .add_key("cb_size", 152)
            .add_key("head_offset", 0)
            .add_key("tensor_keys", "image,mask")
            .build();
        
        let parts: Vec<&str> = handshake.trim().split_whitespace().collect();
        assert_eq!(parts[0], "ZT");
        
        let mut parsed: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        for part in &parts[2..] {
            let (k, v) = part.split_once('=').unwrap();
            parsed.insert(k, v);
        }
        
        assert_eq!(parsed["cb_size"], "152");
        assert_eq!(parsed["head_offset"], "0");
        assert_eq!(parsed["tensor_keys"], "image,mask");
    }
}