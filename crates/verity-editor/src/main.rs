fn main() -> Result<(), Box<dyn std::error::Error>> {
    verity_editor::run(verity_runtime::Game::default())
}
