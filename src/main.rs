use std::{collections::VecDeque, time::{Duration, Instant}};

use bevy::{color::palettes::{basic::*, css::{DARK_BLUE, LIGHT_BLUE, ORANGE}}, platform::collections::HashMap, prelude::*};
use bevy_inspector_egui::{bevy_egui::EguiPlugin, quick::WorldInspectorPlugin};
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
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
            Rotation::East => shape.map(|pos| IVec2::new(pos.y, -pos.x)),
            Rotation::South => shape.map(|pos| IVec2::new(-pos.x, -pos.y)),
            Rotation::West => shape.map(|pos| IVec2::new(-pos.y, pos.x)),
        }
    }
}

#[derive(Resource)]
struct MinoMaterialMap(HashMap<Mino, Handle<ColorMaterial>>);

#[derive(Resource, Default)]
struct MinoMesh{
    default: Handle<Mesh>,
    preview: Handle<Mesh>,
}

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
    fn paralell_move(&mut self, delta: IVec2) {
        self.pivot_pos += delta;
    }
    fn peek_paralell_move(&self, delta: IVec2) -> IVec2 {
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
struct BlockTimer{
    level: u32,
    fall_timer: f32,
    block_landed: bool,
    landed_or_last_operated_time_secs: f32,
    extended_counter: u32,
    min_y: u32,
    is_hard_dropped: bool
}

impl BlockTimer {
    fn new() -> Self {
        BlockTimer {
            level: 1,
            fall_timer: 1.0,
            block_landed: false,
            landed_or_last_operated_time_secs: 0.0,
            extended_counter: 0,
            min_y: ROWS as u32,
            is_hard_dropped: true
        }
    }
    fn new_mino(&mut self, time_secs: f32) {
        self.block_landed = false;
        self.landed_or_last_operated_time_secs = time_secs;
        self.extended_counter = 0;
        self.is_hard_dropped = false;
        self.min_y = ROWS as u32;
    }
    fn falling_interval(&self) -> f32 {
        3.0 / (5 + self.level) as f32
    }
    fn try_fall(&mut self, delta: f32, min_y: u32) -> bool {
        self.fall_timer -= delta;
        if self.fall_timer <= 0.0 {
            self.fall_timer += self.falling_interval();
            if self.min_y > min_y {
                self.extended_counter = 0;
                self.min_y = min_y;
            }
            true
        } else {
            false
        }
    }
    fn block_has_landed(&mut self, time_secs: f32) {
        self.block_landed = true;
        self.landed_or_last_operated_time_secs = time_secs;
    }
    fn hard_drop(&mut self) {
        self.block_landed = true;
        self.is_hard_dropped = true;
    }
    fn not_landed(&mut self) {
        self.block_landed = false;
    }
    fn should_freeze(&mut self, time_secs: f32) -> bool {
        if !self.block_landed {
            return false;
        }
        if self.is_hard_dropped || self.extended_counter >= 15 {
            return true;
        }
        self.landed_or_last_operated_time_secs + 0.5 < time_secs
    }
    fn extend_freeze_time(&mut self, time_secs: f32) {
        self.extended_counter += 1;
        self.landed_or_last_operated_time_secs = time_secs;
    }
}
#[derive(Resource, Default)]
struct LateralMoveTimer {
    move_count: usize,
    last_moved: Option<Instant>,
    is_right: bool
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin::default())
        .add_plugins(WorldInspectorPlugin::new())
        .insert_resource(MinoMaterialMap(HashMap::new()))
        .insert_resource(MinoMesh::default())
        .insert_resource(UpcomingMinoQueue(VecDeque::with_capacity(NEXT_BLOCKS_CAPACITY)))
        .insert_resource(Board([[None; COLUMS]; ROWS + 3]))
        .insert_resource(BlockTimer::new())
        .insert_resource(GameScore::new())
        .insert_resource(HoldedMino::default())
        .insert_resource(LateralMoveTimer::default())
        .init_state::<GameState>()
        .add_systems(Startup, (setup_board_and_resources, setup_score_ui))
        .add_systems(Update, (
            (spawn_mino, move_mino, fall_mino, update_score_ui, hold_mino).run_if(in_state(GameState::Playing)),
            updata_preview_minos.run_if(resource_changed::<HoldedMino>.or(resource_changed::<UpcomingMinoQueue>)),
        ))
        .add_systems(PostUpdate, (
            (freeze_block, clear_lines).chain().run_if(in_state(GameState::Playing)),
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
    // rgba(221, 213, 193, 1)
    let grid_line_material = materials.add(Color::srgb_u8(221, 213, 193));
    let row_line_mesh = meshes.add(Rectangle::new((COLUMS as f32 + 0.1) * UNIT, 0.1* UNIT));
    let col_line_mesh = meshes.add(Rectangle::new(0.1 * UNIT, (ROWS as f32 + 0.1) * UNIT));
    for row in 0..=ROWS {
        commands.spawn((
            Mesh2d(row_line_mesh.clone()),
            MeshMaterial2d(grid_line_material.clone()),
            Transform::from_translation(Vec3::new(0.0, - (ROWS as f32 / 2.0) * UNIT + row as f32 * UNIT, -0.5)),
        ));
    }
    for col in 0..=COLUMS {
        commands.spawn((
            Mesh2d(col_line_mesh.clone()),
            MeshMaterial2d(grid_line_material.clone()),
            Transform::from_translation(Vec3::new(- (COLUMS as f32 / 2.0) * UNIT + col as f32 * UNIT, 0.0, -0.5)),
        ));
    }


    // ブロックの色とメッシュを準備
    let mut map = HashMap::new();
    map.insert(Mino::O, materials.add(Color::from(YELLOW)));
    map.insert(Mino::T, materials.add(Color::from(PURPLE)));
    map.insert(Mino::S, materials.add(Color::from(GREEN)));
    map.insert(Mino::Z, materials.add(Color::from(RED)));
    map.insert(Mino::L, materials.add(Color::from(ORANGE)));
    map.insert(Mino::J, materials.add(Color::from(DARK_BLUE)));
    map.insert(Mino::I, materials.add(Color::from(LIGHT_BLUE)));
    block_material_map.0 = map;

    mino_mesh.default = meshes.add(Rectangle::new(UNIT * 0.9, UNIT * 0.9));
    mino_mesh.preview = meshes.add(Rectangle::new(UNIT * 0.45, UNIT * 0.45));
}

fn spawn_mino(
    mut commands: Commands,
    block_color_map: Res<MinoMaterialMap>,
    mino_mesh: Res<MinoMesh>,
    mut next_blocks: ResMut<UpcomingMinoQueue>,
    control_block: Query<&ControllingBlock>,
    board: Res<Board>,
    mut next_state: ResMut<NextState<GameState>>,
    mut holded_mino: ResMut<HoldedMino>,
    mut block_timer: ResMut<BlockTimer>,
    time: Res<Time<Virtual>>
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
                Mesh2d(mino_mesh.default.clone()),
                MeshMaterial2d(material.clone()),
                Transform::from_translation(world_pos),
                block,
            ));
        }
        // ホールド可能にする
        holded_mino.can_hold = true;
        // block_timerのリセット
        block_timer.new_mino(time.elapsed_secs());
    }
}

fn fall_mino(
    time: Res<Time>,
    mut block_timer: ResMut<BlockTimer>,
    mut control_block: Query<(&mut Transform, &mut ControllingBlock)>,
    board: Res<Board>,
    keys: Res<ButtonInput<KeyCode>>,
    mut game_score: ResMut<GameScore>,
){
    if control_block.is_empty() {
        return;
    }
    let can_fall = control_block.iter().all(|(_, block)| {
        let new_pos = block.peek_paralell_move(IVec2::new(0, -1));
        board.is_position_valid(new_pos)
    });

    if !can_fall && !block_timer.block_landed {
        block_timer.block_has_landed(time.elapsed_secs());
    }else if can_fall && block_timer.block_landed {
        block_timer.not_landed();
    }

    if !can_fall {
        return;
    }

    let delta = if keys.pressed(KeyCode::ArrowDown) || keys.pressed(KeyCode::KeyS) {
        // ソフトドロップ
        game_score.score += 1;
        time.delta_secs() * 20.0
    } else {
        time.delta_secs()
    };

    let min_y = control_block.iter().map(|(_, block)|
        block.peek_paralell_move(IVec2::new(0, -1)).y
    ).min().unwrap();

    if !block_timer.try_fall(delta, min_y as u32) {
        return;
    }

    for (mut transform, mut block) in control_block.iter_mut() {
        block.paralell_move(IVec2::new(0, -1));
        transform.translation = grid_to_world_position(block.get_board_position());
    }
}

fn move_mino(
    keys: Res<ButtonInput<KeyCode>>,
    mut control_block: Query<(&mut Transform, &mut ControllingBlock)>,
    board: Res<Board>,
    mut block_timer: ResMut<BlockTimer>,
    mut game_score: ResMut<GameScore>,
    mut lateral_move_timer: ResMut<LateralMoveTimer>,
    time: Res<Time>
){
    let mut is_operated = false;
    if keys.any_just_pressed([KeyCode::ArrowRight, KeyCode::KeyD, KeyCode::ArrowLeft, KeyCode::KeyA]) {
        lateral_move_timer.move_count = 0;
        lateral_move_timer.last_moved = Some(Instant::now());
        lateral_move_timer.is_right = keys.any_just_pressed([KeyCode::ArrowRight, KeyCode::KeyD]);
    }
    else if !keys.any_pressed([KeyCode::ArrowRight, KeyCode::KeyD, KeyCode::ArrowLeft, KeyCode::KeyA]) {
        lateral_move_timer.last_moved = None;
    }
    let first_interval = Duration::from_millis(300);
    let next_interval = Duration::from_millis(50);
    // 横移動
    let lateral_delta = match &mut *lateral_move_timer {
        LateralMoveTimer {
            last_moved: Some(last),
            move_count,
            is_right
        } if *move_count == 0 || last.elapsed() > if *move_count == 1 { first_interval } else { next_interval } => {
            *last = Instant::now();
            *move_count += 1;
            IVec2::new(if *is_right { 1 } else { -1 }, 0)
        }
        _ => IVec2::new(0, 0)
    };

    if lateral_delta != IVec2::new(0, 0) 
        && control_block.iter().all(|(_, block)| board.is_position_valid(block.peek_paralell_move(lateral_delta))) {
        for (mut transform, mut block) in control_block.iter_mut() {
            block.paralell_move(lateral_delta);
            transform.translation = grid_to_world_position(block.get_board_position());
            is_operated = true;
        }
    }
    if is_operated {
        block_timer.extend_freeze_time(time.elapsed_secs());
    }
    is_operated = false;

    // ハードドロップ
    if keys.just_pressed(KeyCode::Space) {
        let drop_distance = control_block.iter().map(|(_, block)| 
            (0..).find(|&d| !board.is_position_valid(block.peek_paralell_move(IVec2::new(0, -(d + 1))))).unwrap()
        ).min().unwrap();
        block_timer.hard_drop();
        game_score.score += drop_distance as u32 * 5;
        for (mut transform, mut block) in control_block.iter_mut() {
            block.paralell_move(IVec2::new(0, - (drop_distance as i32)));
            transform.translation = grid_to_world_position(block.get_board_position());
        }
    }

    let (mino_kind, mino_rotation) = if let Some((_, block)) = control_block.iter().next() {
        (block.kind, block.rotation)
    } else {
        return;
    };

    // 回転
    let rotate_right = keys.just_pressed(KeyCode::ArrowUp) || keys.just_pressed(KeyCode::KeyW);
    let rotate_left = keys.just_pressed(KeyCode::KeyQ) || keys.just_pressed(KeyCode::KeyZ);

    if rotate_right == rotate_left {
        return;
    }
    let next_rotation = if rotate_right { mino_rotation.rotate_right() } else { mino_rotation.rotate_left() };
    
    // Super Rotation Systemでの回転軸の補正順序
    let mut wall_kick_offsets;
    if mino_kind != Mino::I {
        wall_kick_offsets = [
            IVec2::new(0, 0),
            IVec2::new(-1, 0),
            IVec2::new(-1, 1),
            IVec2::new(0, -2),
            IVec2::new(-1, -2),
        ];
        if mino_rotation == Rotation::East || next_rotation == Rotation::West {
            for offset in &mut wall_kick_offsets {
                offset.x *= -1;
            }
        }
        if matches!(mino_rotation, Rotation::East | Rotation::West) {
            for offset in &mut wall_kick_offsets {
                offset.y *= -1;
            }
        }
    }
    else { // Iミノの場合のみ別の補正順序
        wall_kick_offsets = [
            IVec2::new(0,0),
            IVec2::new(-2,0),
            IVec2::new(1,0),
            IVec2::new(-2,-1),
            IVec2::new(1,2),
        ];
        let mut rotation_pair = [mino_rotation, next_rotation];
        rotation_pair.sort();
        if matches!(rotation_pair, [Rotation::East, Rotation::South] | [Rotation::North, Rotation::West]) {
            wall_kick_offsets.swap(1, 2);
            wall_kick_offsets.swap(3, 4);
        }
        if mino_rotation == Rotation::East || next_rotation == Rotation::West {
            for offset in &mut wall_kick_offsets {
                offset.x *= -1;
            }
        }
        if mino_rotation == Rotation::South || next_rotation == Rotation::North {
            for offset in &mut wall_kick_offsets {
                offset.y *= -1;
            }
        }
    }

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
            block.paralell_move(offset);
            transform.translation = grid_to_world_position(block.get_board_position());
            is_operated = true;
        }
    }

    // 凍結タイマー延長
    if is_operated {
        block_timer.extend_freeze_time(time.elapsed_secs());   
    }
}

#[derive(Resource, Default)]
struct HoldedMino{
    mino: Option<Mino>,
    can_hold: bool,
}

fn hold_mino(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut holded_mino: ResMut<HoldedMino>,
    mut control_block: Query<(Entity, &mut Transform, &mut ControllingBlock)>,
    block_color_map: Res<MinoMaterialMap>,
    mino_mesh: Res<MinoMesh>,
){
    // 一度ホールドしたら次のブロックがスポーンするまでホールドできない
    if !holded_mino.can_hold {
        return;
    }
    // ホールドキーが押されたらホールド処理
    if !keys.just_pressed(KeyCode::KeyC) {
        return;
    }
    let mino_to_hold = if let Some((_, _, block)) = control_block.iter_mut().next() {
        block.kind
    } else {
        return;
    };
    // ホールドしているブロックと入れ替え
    let new_mino = holded_mino.mino;
    holded_mino.mino = Some(mino_to_hold);
    holded_mino.can_hold = false;
    for (entity, _, _) in control_block.iter_mut() {
        commands.entity(entity).despawn();
    }
    if let Some(new) = new_mino {
        // ホールドしているブロックをスポーン
        let material = block_color_map.0.get(&new).unwrap().clone();
        for i in 0..4 {
            let block = ControllingBlock{
                kind: new,
                rotation: Rotation::North,
                index_in_mino: i,
                pivot_pos: IVec2::new(COLUMS as i32 / 2, ROWS as i32 - 1),
            };
            let world_pos = grid_to_world_position(block.get_board_position());
            commands.spawn((
                Name::new(format!("Mino {:?} - Block {}", new, i)),
                Mesh2d(mino_mesh.default.clone()),
                MeshMaterial2d(material.clone()),
                Transform::from_translation(world_pos),
                block,
            ));
        }
    }
}

#[derive(Component)]
struct PreviewMino;

fn updata_preview_minos(
    mut commands: Commands,
    preview_mino_query: Query<Entity, With<PreviewMino>>,
    block_color_map: Res<MinoMaterialMap>,
    mino_mesh: Res<MinoMesh>,
    holded_mino: Res<HoldedMino>,
    next_blocks: Res<UpcomingMinoQueue>,
){
    // 既存のプレビューを削除
    for entity in preview_mino_query.iter() {
        commands.entity(entity).despawn();
    }
    // ホールドされたミノは左上に表示
    if let Some(holded) = holded_mino.mino {
        let material = block_color_map.0.get(&holded).unwrap().clone();
        for &pos in holded.get_rotated_shape(&Rotation::North).iter() {
            let world_pos = Vec3::new(
                - (COLUMS as f32 / 2.0 + 2.0) * UNIT + pos.x as f32 * UNIT * 0.5,
                (ROWS as f32 / 2.0 - 2.0) * UNIT + pos.y as f32 * UNIT * 0.5,
                0.0,
            );
            commands.spawn((
                Name::new(format!("Preview Holded Mino {:?} - Block", holded)),
                Mesh2d(mino_mesh.preview.clone()),
                MeshMaterial2d(material.clone()),
                Transform::from_translation(world_pos),
                PreviewMino,
            ));
        }
    }
    // 次のミノは右上に表示
    for (i, &mino) in next_blocks.0.iter().take(NEXT_BLOCKS_CAPACITY).enumerate() {
        let material = block_color_map.0.get(&mino).unwrap().clone();
        for &pos in mino.get_rotated_shape(&Rotation::North).iter() {
            let world_pos = Vec3::new(
                (COLUMS as f32 / 2.0 + 2.0) * UNIT + pos.x as f32 * UNIT * 0.5,
                (ROWS as f32 / 2.0 - 2.0 - i as f32 * 3.0) * UNIT + pos.y as f32 * UNIT * 0.5,
                0.0,
            );
            commands.spawn((
                Name::new(format!("Preview Next Mino {:?} - Block", mino)),
                Mesh2d(mino_mesh.preview.clone()),
                MeshMaterial2d(material.clone()),
                Transform::from_translation(world_pos),
                PreviewMino,
            ));
        }
    }
}

fn freeze_block(
    mut commands: Commands,
    time: Res<Time<Virtual>>,
    mut block_timer: ResMut<BlockTimer>,
    control_block: Query<(Entity, &ControllingBlock)>,
    mut board: ResMut<Board>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if !block_timer.should_freeze(time.elapsed_secs()) {
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
    mut block_timer: ResMut<BlockTimer>,
    mut game_score: ResMut<GameScore>,
    mut frozen_blocks: Query<(&mut Transform, &mut FrozenBlock)>,
){
    let mut num_erased_lines = 0;
    for row in 0..ROWS {
        if board.0[row].iter().all(|cell| cell.is_some()) {
            // 行を消す
            for &cell in board.0[row].iter() {
                if let Some(entity) = cell {
                    commands.entity(entity).despawn();
                }
            }
            num_erased_lines += 1;
            // 上の行を一つ下にずらす
            for r in row + 1..ROWS + 3 {
                for c in 0..COLUMS {
                    board.0[r - 1][c] = board.0[r][c];
                    if let Some(entity) = board.0[r][c] 
                        && let Ok((mut transform, mut block)) = frozen_blocks.get_mut(entity)
                    {
                        block.pos.y -= 1;
                        transform.translation = grid_to_world_position(block.pos);
                    }
                }
            }
            // 一番上の行を空にする
            for c in 0..COLUMS {
                board.0[ROWS + 2][c] = None;
            }
        }
    }
    game_score.add_erased_lines(num_erased_lines);
    if num_erased_lines > 0 {
        block_timer.level = game_score.level;
    }
}

#[derive(Resource)]
struct GameScore{
    score: u32,
    level: u32,
    erased_lines: u32,
    combo_count: u32,
}

impl GameScore {
    fn new() -> Self {
        GameScore {
            score: 0,
            level: 1,
            erased_lines: 0,
            combo_count: 0,
        }
    }
    fn add_erased_lines(&mut self, num_lines: u32) {
        if num_lines == 0 {
            self.combo_count = 0;
            return;
        }
        self.erased_lines += num_lines;
        self.combo_count += 1;
        self.score += match num_lines {
            1 => 100,
            2 => 300,
            3 => 500,
            4 => 800,
            _ => 0,
        } * self.combo_count;
        self.level = self.erased_lines / 10 + 1;
    }
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
            Name::new("Labels"),
            Text::new("Score\nLevel\nLines"),
            text_font.clone(),
            TextColor(WHITE.into()),
            TextLayout::new_with_justify(Justify::Left),
        ));
        builder.spawn((
            Name::new("Separators"),
            Text::new(":\n:\n:"),
            text_font.clone(),
            TextColor(GRAY.into()),
            TextLayout::new_with_justify(Justify::Center),
        ));
        builder.spawn((
            Name::new("Values"),
            Node {
                flex_direction: FlexDirection::Column,
                flex_grow: 1.0,
                ..default()
            },
        )).with_children(|parent| {
            parent.spawn((
                Name::new("ScoreValue"),
                ScoreText,
                Text::new("0"),
                text_font.clone(),
                TextColor(WHITE.into()),
                TextLayout::new_with_justify(Justify::Left),
            ));
            parent.spawn((
                Name::new("LevelValue"),
                LevelText,
                Text::new("1"),
                text_font.clone(),
                TextColor(WHITE.into()),
                TextLayout::new_with_justify(Justify::Left),
            ));
            parent.spawn((
                Name::new("ElasedLinesValue"),
                ElasedLinesText,
                Text::new("0"),
                text_font.clone(),
                TextColor(WHITE.into()),
                TextLayout::new_with_justify(Justify::Left),
            ));
        });
    });
}

fn update_score_ui(
    game_score: Res<GameScore>,
    mut score_text: Query<&mut Text, (With<ScoreText>, Without<LevelText>, Without<ElasedLinesText>)>,
    mut level_text: Query<&mut Text, (With<LevelText>, Without<ScoreText>, Without<ElasedLinesText>)>,
    mut elased_lines_text: Query<&mut Text, (With<ElasedLinesText>, Without<ScoreText>, Without<LevelText>)>,
){
    if let Ok(mut text_span) = score_text.single_mut() {
        **text_span = format!("{}", game_score.score);
    }
    if let Ok(mut text_span) = level_text.single_mut() {
        **text_span = format!("{}", game_score.level);
    }
    if let Ok(mut text_span) = elased_lines_text.single_mut() {
        **text_span = format!("{}", game_score.erased_lines);
    }
}