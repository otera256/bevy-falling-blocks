use std::collections::VecDeque;

use bevy::{color::palettes::{basic::*, css::{DARK_BLUE, LIGHT_BLUE, ORANGE}}, platform::collections::HashMap, prelude::*};
use bevy_inspector_egui::{bevy_egui::EguiPlugin, quick::WorldInspectorPlugin};
use itertools::{Itertools, iproduct};
use rand::seq::IndexedRandom;

const UNIT: f32 = 35.0;
const ROWS: usize = 20;
const COLUMS: usize = 10;

fn grid_to_world_position(grid_pos: IVec2) -> Vec3 {
    let origin = Vec3::new(
        - ((COLUMS - 1) as f32) * UNIT / 2.0,
        - ((ROWS - 1) as f32) * UNIT / 2.0,
        0.0,
    );
    Vec3::new(
        origin.x + grid_pos.x as f32 * UNIT,
        origin.y + grid_pos.y as f32 * UNIT,
        0.0,
    )
}

#[derive(States, Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
enum GameState{
    #[default]
    Playing,
    GameOver
}

#[derive(Component, Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Mino{
    O,
    T,
    S,
    Z,
    L,
    J,
    I
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Rotation{
    North,
    East,
    South,
    West,
}

impl Rotation {
    fn rotate_right(&self) -> Rotation {
        match self {
            Rotation::North => Rotation::East,
            Rotation::East => Rotation::South,
            Rotation::South => Rotation::West,
            Rotation::West => Rotation::North,
        }
    }
    fn rotate_left(&self) -> Rotation {
        match self {
            Rotation::North => Rotation::West,
            Rotation::West => Rotation::South,
            Rotation::South => Rotation::East,
            Rotation::East => Rotation::North,
        }
    }
}

impl Mino{
    fn shape(&self) -> [IVec2; 4] {
        match self {
            // □□
            // ■□
            Mino::O => [IVec2::new(0,0), IVec2::new(1,0), IVec2::new(0,1), IVec2::new(1,1)],
            //  □
            // □■□
            Mino::T => [IVec2::new(-1,0), IVec2::new(0,0), IVec2::new(1,0), IVec2::new(0,1)],
            //  □□
            // □■
            Mino::S => [IVec2::new(-1,0), IVec2::new(0,0), IVec2::new(0,1), IVec2::new(1,1)],
            // □□
            //  ■□
            Mino::Z => [IVec2::new(-1,1), IVec2::new(0,1), IVec2::new(0,0), IVec2::new(1,0)],
            //   □
            // □■□
            Mino::L => [IVec2::new(-1,0), IVec2::new(0,0), IVec2::new(1,0), IVec2::new(1,1)],
            // □
            // □■□
            Mino::J => [IVec2::new(-1,1), IVec2::new(-1,0), IVec2::new(0,0), IVec2::new(1,0)],
            // □□□□
            Mino::I => [IVec2::new(-2,0), IVec2::new(-1,0), IVec2::new(0,0), IVec2::new(1,0)],
        }
    }
    fn get_rotated_shape(&self, rotation: &Rotation) -> [IVec2; 4] {
        let shape = self.shape();
        if *self == Mino::O {
            return shape;
        }else if *self == Mino::I {
            return match rotation {
                Rotation::North => shape,
                Rotation::East => shape.map(|pos| IVec2::new(0, -pos.x - 1)),
                Rotation::South => shape.map(|pos| IVec2::new(-pos.x - 1, -1)),
                Rotation::West => shape.map(|pos| IVec2::new(-1, pos.x)),
            }
        }
        match rotation {
            Rotation::North => shape,
            Rotation::East => shape.map(|pos| IVec2::new(-pos.y, pos.x)),
            Rotation::South => shape.map(|pos| IVec2::new(-pos.x, -pos.y)),
            Rotation::West => shape.map(|pos| IVec2::new(pos.y, -pos.x)),
        }
    }
}

#[derive(Resource)]
struct MinoMaterialMap(HashMap<Mino, Handle<ColorMaterial>>);

#[derive(Resource)]
struct MinoMesh(Handle<Mesh>);

const NEXT_BLOCKS_CAPACITY: usize = 5;

#[derive(Resource)]
struct UpcomingMinoQueue(VecDeque<Mino>);

#[derive(Component)]
struct ControllingBlock{
    kind: Mino,
    rotation: Rotation,
    index_in_mino: usize,
    pivot_pos: IVec2,
}

impl ControllingBlock {
    fn get_board_position(&self) -> IVec2 {
        self.pivot_pos + self.kind.get_rotated_shape(&self.rotation)[self.index_in_mino]
    }
    fn parrallel_move(&mut self, delta: IVec2) {
        self.pivot_pos += delta;
    }
    fn peek_parrallel_move(&self, delta: IVec2) -> IVec2 {
        self.get_board_position() + delta
    }
    fn rotate(&mut self, rotation: Rotation) {
        self.rotation = rotation;
    }
    fn peek_rotate(&self, rotation: Rotation) -> IVec2 {
        let rotated_shape = self.kind.get_rotated_shape(&rotation);
        self.pivot_pos + rotated_shape[self.index_in_mino]
    }
}

#[derive(Component)]
struct FrozenBlock{
    pos: IVec2,
}

#[derive(Resource)]
struct Board([[Option<Entity>; COLUMS]; ROWS + 3]);

impl Board {
    // 上には3つまではみ出られる
    fn is_position_valid(&self, pos: IVec2) -> bool {
        if pos.x < 0 || pos.x >= COLUMS as i32 || pos.y < 0 || pos.y >= (ROWS + 3) as i32 {
            return false;
        }
        self.0[pos.y as usize][pos.x as usize].is_none()
    }
    fn is_position_rigidly_valid(&self, pos: IVec2) -> bool {
        if pos.x < 0 || pos.x >= COLUMS as i32 || pos.y < 0 || pos.y >= ROWS as i32 {
            return false;
        }
        self.0[pos.y as usize][pos.x as usize].is_none()
    }
}

#[derive(Resource)]
struct GameLevel{
    level: u32,
    erased_lines: u32,
    fall_timer: f32,
    block_landed: bool,
    freeze_timer: f32,
}

impl GameLevel {
    fn new() -> Self {
        GameLevel {
            level: 1,
            erased_lines: 0,
            fall_timer: 1.0,
            block_landed: false,
            freeze_timer: 1.0,
        }
    }
    fn falling_interval(&self) -> f32 {
        3.0 / ((5 + self.level) as f32).sqrt()
    }
    fn add_erased_lines(&mut self, lines: u32) {
        self.erased_lines += lines;
        self.level = self.erased_lines / 10 + 1;
    }
    fn should_fall(&mut self, delta: f32) -> bool {
        self.fall_timer -= delta;
        if self.fall_timer <= 0.0 {
            self.fall_timer += self.falling_interval();
            true
        } else {
            false
        }
    }
    fn block_has_landed(&mut self) {
        self.block_landed = true;
        self.freeze_timer = self.falling_interval();
    }
    fn hard_drop(&mut self) {
        self.block_landed = true;
        self.freeze_timer = 0.0;
    }
    fn block_reset(&mut self) {
        self.block_landed = false;
    }
    fn should_freeze(&mut self, delta: f32) -> bool {
        if !self.block_landed {
            return false;
        }
        self.freeze_timer -= delta;
        if self.freeze_timer <= 0.0 {
            true
        } else {
            false
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin::default())
        .add_plugins(WorldInspectorPlugin::new())
        .insert_resource(MinoMaterialMap(HashMap::new()))
        .insert_resource(MinoMesh(Handle::default()))
        .insert_resource(UpcomingMinoQueue(VecDeque::with_capacity(NEXT_BLOCKS_CAPACITY)))
        .insert_resource(Board([[None; COLUMS]; ROWS + 3]))
        .insert_resource(GameLevel::new())
        .init_state::<GameState>()
        .add_systems(Startup, (setup_board_and_resources, setup_score_ui))
        .add_systems(Update, 
            (spawn_mino, move_mino, fall_mino, update_score_ui).run_if(in_state(GameState::Playing))
        )
        .add_systems(PostUpdate, (
            (freeze_block, clear_lines).run_if(in_state(GameState::Playing)),
        ))
        .run();
}

fn setup_board_and_resources(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut block_material_map: ResMut<MinoMaterialMap>,
    mut mino_mesh: ResMut<MinoMesh>,
){
    commands.spawn(Camera2d);

    // rgba(231, 226, 213, 1)
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(COLUMS as f32 * UNIT, ROWS as f32 * UNIT))),
        MeshMaterial2d(materials.add(Color::srgb_u8(231, 226, 213))),
        Transform::from_xyz(0.0, 0.0, -1.0),
    ));

    // setuup block materials
    let mut map = HashMap::new();
    map.insert(Mino::O, materials.add(Color::from(YELLOW)));
    map.insert(Mino::T, materials.add(Color::from(PURPLE)));
    map.insert(Mino::S, materials.add(Color::from(GREEN)));
    map.insert(Mino::Z, materials.add(Color::from(RED)));
    map.insert(Mino::L, materials.add(Color::from(ORANGE)));
    map.insert(Mino::J, materials.add(Color::from(DARK_BLUE)));
    map.insert(Mino::I, materials.add(Color::from(LIGHT_BLUE)));
    block_material_map.0 = map;

    mino_mesh.0 = meshes.add(Rectangle::new(UNIT * 0.9, UNIT * 0.9));
}

fn spawn_mino(
    mut commands: Commands,
    block_color_map: Res<MinoMaterialMap>,
    mino_mesh: Res<MinoMesh>,
    mut next_blocks: ResMut<UpcomingMinoQueue>,
    control_block: Query<&ControllingBlock>,
    board: Res<Board>,
    mut next_state: ResMut<NextState<GameState>>,
){
    // NEXT_BLOCKS_CAPACITYまでブロックを補充
    let all_blocks = [Mino::O, Mino::T, Mino::S, Mino::Z, Mino::L, Mino::J, Mino::I];
    let mut rng = rand::rng();
    while next_blocks.0.len() < NEXT_BLOCKS_CAPACITY {
        if let Some(&block) = all_blocks.choose(&mut rng) {
            next_blocks.0.push_back(block);
        }
    }

    // 次のブロックを取り出してスポーン
    if control_block.is_empty() && let Some(mino) = next_blocks.0.pop_front() {
        info!("Spawning mino: {:?}", mino);
        let material = block_color_map.0.get(&mino).unwrap().clone();
        // もしスポーンする位置にブロックがあったらゲームオーバー
        for i in 0..4 {
            let block = ControllingBlock{
                kind: mino,
                rotation: Rotation::North,
                index_in_mino: i,
                pivot_pos: IVec2::new(COLUMS as i32 / 2, ROWS as i32 - 1),
            };
            if !board.is_position_valid(block.get_board_position()) {
                info!("Game Over!");
                next_state.set(GameState::GameOver);
                return;
            }
            let world_pos = grid_to_world_position(block.get_board_position());
            commands.spawn((
                Name::new(format!("Mino {:?} - Block {}", mino, i)),
                Mesh2d(mino_mesh.0.clone()),
                MeshMaterial2d(material.clone()),
                Transform::from_translation(world_pos),
                block,
            ));
        }
    }
}

fn fall_mino(
    time: Res<Time>,
    mut game_level: ResMut<GameLevel>,
    mut control_block: Query<(&mut Transform, &mut ControllingBlock)>,
    board: Res<Board>,
    keys: Res<ButtonInput<KeyCode>>,
){
    let can_fall = control_block.iter().all(|(_, block)| {
        let new_pos = block.peek_parrallel_move(IVec2::new(0, -1));
        board.is_position_valid(new_pos)
    });

    if !can_fall && !game_level.block_landed {
        game_level.block_has_landed();
    }else if can_fall && game_level.block_landed {
        game_level.block_reset();
    }

    if !can_fall {
        return;
    }

    let delta = if keys.pressed(KeyCode::ArrowDown) || keys.pressed(KeyCode::KeyS) {
        // ソフトドロップ
        time.delta_secs() * 10.0
    } else {
        time.delta_secs()
    };
    if !game_level.should_fall(delta) {
        return;
    }

    for (mut transform, mut block) in control_block.iter_mut() {
        block.parrallel_move(IVec2::new(0, -1));
        transform.translation = grid_to_world_position(block.get_board_position());
    }
}

fn move_mino(
    keys: Res<ButtonInput<KeyCode>>,
    mut control_block: Query<(&mut Transform, &mut ControllingBlock)>,
    board: Res<Board>,
    mut game_level: ResMut<GameLevel>
){
    // 横移動
    let lateral_delta = if keys.just_pressed(KeyCode::ArrowRight) || keys.just_pressed(KeyCode::KeyD) {
        IVec2::new(1, 0)
    } else if keys.just_pressed(KeyCode::ArrowLeft) || keys.just_pressed(KeyCode::KeyA) {
        IVec2::new(-1, 0)
    } else {
        IVec2::new(0, 0)
    };
    if lateral_delta != IVec2::new(0, 0) 
        && control_block.iter().all(|(_, block)| board.is_position_valid(block.peek_parrallel_move(lateral_delta))) {
        for (mut transform, mut block) in control_block.iter_mut() {
            block.parrallel_move(lateral_delta);
            transform.translation = grid_to_world_position(block.get_board_position());
        }
    }

    // ハードドロップ
    if keys.just_pressed(KeyCode::Space) {
        let drop_distance = control_block.iter().map(|(_, block)| 
            (0..).find(|&d| !board.is_position_valid(block.peek_parrallel_move(IVec2::new(0, -(d + 1))))).unwrap()
        ).min().unwrap();
        game_level.hard_drop();
        for (mut transform, mut block) in control_block.iter_mut() {
            block.parrallel_move(IVec2::new(0, - (drop_distance as i32)));
            transform.translation = grid_to_world_position(block.get_board_position());
        }
    }

    // 回転
    // Super Rotation Systemでの回転軸の補正順序（右回転の場合）
    // 左回転の場合はx座標を反転させる
    let wall_kick_offsets = [
        IVec2::new(0, 0),
        IVec2::new(-1, 0),
        IVec2::new(-1, 1),
        IVec2::new(0, -2),
        IVec2::new(-1, -2),
    ];
    let mino_rotaion = if let Some((_, block)) = control_block.iter().next() {
        block.rotation
    } else {
        return;
    };
    let (next_rotation, wall_kick_offsets) = if keys.just_pressed(KeyCode::ArrowUp) || keys.just_pressed(KeyCode::KeyW) {
        (mino_rotaion.rotate_right(), wall_kick_offsets)
    } else if keys.just_pressed(KeyCode::KeyQ) {
        (mino_rotaion.rotate_left(), wall_kick_offsets.map(|offset| IVec2::new(-offset.x, offset.y)))
    } else {
        return;
    };
    // wall_kick_offsetsを順に試して、どれかで回転できたら回転を適用
    // できなかったら回転しない
    let can_rotate = wall_kick_offsets.iter().find_map(|&offset| {
        if control_block.iter().all(|(_, block)| {
            let new_pos = block.peek_rotate(next_rotation) + offset;
            board.is_position_valid(new_pos)
        }) {
            Some(offset)
        } else {
            None
        }
    });
    if let Some(offset) = can_rotate {
        for (mut transform, mut block) in control_block.iter_mut() {
            block.rotate(next_rotation);
            block.parrallel_move(offset);
            transform.translation = grid_to_world_position(block.get_board_position());
        }
    }
}

fn freeze_block(
    mut commands: Commands,
    time: Res<Time>,
    mut game_level: ResMut<GameLevel>,
    control_block: Query<(Entity, &ControllingBlock)>,
    mut board: ResMut<Board>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if !game_level.should_freeze(time.delta_secs()) {
        return;
    }
    // もしブロックが一つもROW * COLUMSに収まらなかったらゲームオーバー
    if control_block.iter().any(|(_, block)| {
        !board.is_position_rigidly_valid(block.get_board_position())
    }) {
        info!("Game Over!");
        next_state.set(GameState::GameOver);
        return;
    }
    for (entity, block) in control_block.iter() {
        let board_pos = block.get_board_position();
        board.0[board_pos.y as usize][board_pos.x as usize] = Some(entity);
        commands.entity(entity)
            .remove::<ControllingBlock>()
            .insert(FrozenBlock{ pos: board_pos });
    }
}

fn clear_lines(
    mut commands: Commands,
    mut board: ResMut<Board>,
    mut game_level: ResMut<GameLevel>,
    mut frozen_blocks: Query<(&mut Transform, &mut FrozenBlock)>,
){
    let lines_to_erase = (0..ROWS).filter(|&y| {
        (0..COLUMS).all(|x| board.0[y][x].is_some())
    }).collect_vec();
    let num_erased_lines = lines_to_erase.len() as u32;
    if num_erased_lines == 0 {
        return;
    }
    // ブロックを消す
    for (x, &y) in iproduct!(0..COLUMS, &lines_to_erase) {
        if let Some(entity) = board.0[y][x] {
            commands.entity(entity).despawn();
            board.0[y][x] = None;
        }
    }
    // ブロックを落とす
    for &y in &lines_to_erase {
        for (x, yy) in iproduct!(0..COLUMS, (y + 1)..(ROWS + 3)) {
            if let Some(entity) = board.0[yy][x] {
                board.0[yy - 1][x] = Some(entity);
                board.0[yy][x] = None;
                if let Ok((mut transform, mut frozen_block)) = frozen_blocks.get_mut(entity) {
                    frozen_block.pos.y -= 1;
                    transform.translation = grid_to_world_position(frozen_block.pos);
                }
            }
        }
    }
    game_level.add_erased_lines(num_erased_lines);
}

#[derive(Component)]
struct ScoreText;

#[derive(Component)]
struct LevelText;

#[derive(Component)]
struct ElasedLinesText;

fn setup_score_ui(
    mut commands: Commands,
){
    let text_font = TextFont{
        font_size: 32.0,
        ..default()
    };
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Start,
            justify_content: JustifyContent::Start,
            column_gap: px(20),
            bottom: px(20),
            left: px(20),
            ..default()
        },
    )).with_children(|builder| {
        builder.spawn((
            Text::new("Score\nLevel\nLines"),
            text_font.clone(),
            TextColor(WHITE.into()),
            TextLayout::new_with_justify(Justify::Left),
        ));
        builder.spawn((
            Text::new(":\n:\n:"),
            text_font.clone(),
            TextColor(GRAY.into()),
            TextLayout::new_with_justify(Justify::Center),
        ));
        builder.spawn((
            Node {
                flex_direction: FlexDirection::Column,
                ..default()
            },
            children![(
                ScoreText,
                TextSpan::new("0"),
                text_font.clone(),
                TextColor(WHITE.into()),
                TextLayout::new_with_justify(Justify::Left),
            ), (
                LevelText,
                TextSpan::new("1"),
                text_font.clone(),
                TextColor(WHITE.into()),
                TextLayout::new_with_justify(Justify::Left),
            ), (
                ElasedLinesText,
                TextSpan::new("0"),
                text_font.clone(),
                TextColor(WHITE.into()),
                TextLayout::new_with_justify(Justify::Left),
            )]
        ));
    });
}

fn update_score_ui(
    game_level: Res<GameLevel>,
    mut score_text: Query<&mut TextSpan, (With<ScoreText>, Without<LevelText>, Without<ElasedLinesText>)>,
    mut level_text: Query<&mut TextSpan, (With<LevelText>, Without<ScoreText>, Without<ElasedLinesText>)>,
    mut elased_lines_text: Query<&mut TextSpan, (With<ElasedLinesText>, Without<ScoreText>, Without<LevelText>)>,
){
    if let Ok(mut text_span) = score_text.single_mut() {
        *text_span = TextSpan::new(format!("{}", game_level.erased_lines * 100));
    }
    if let Ok(mut text_span) = level_text.single_mut() {
        *text_span = TextSpan::new(format!("{}", game_level.level));
    }
    if let Ok(mut text_span) = elased_lines_text.single_mut() {
        *text_span = TextSpan::new(format!("{}", game_level.erased_lines));
    }
}