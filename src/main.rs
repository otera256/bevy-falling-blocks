use bevy::prelude::*;

const UNIT: f32 = 40.0;

fn main() {
    println!("Hello, world!");
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
){
    commands.spawn(Camera2d);

    // rgba(231, 226, 213, 1)
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(10.0 * UNIT, 20.0 * UNIT))),
        MeshMaterial2d(materials.add(Color::srgb_u8(231, 226, 213)))
    ));
}