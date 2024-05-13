Date: 2023.1.30 
crystal_value:5
root_mode: "survival" (no need to wait 5 min after entering the server)

# Modding code -----------------------------------------------------------------

var map = 
"1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n"+
"                                           \n"+
"                                           \n"+
"2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2\n"+
"                                           \n"+
"                                           \n"+
"3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3\n"+
"                                           \n"+
"                                           \n"+
"4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4\n"+
"                                           \n"+
"                                           \n"+
"5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5\n"+
"                                           \n"+
"                                           \n"+
"6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6\n"+
"                                           \n"+
"                                           \n"+
"7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7\n"+
"                                           \n"+
"                                           \n"+
"8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8\n"+
"                                           \n"+
"                                           \n"+
"9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9\n";


this.options = {
  root_mode: "survival",
  custom_map: map,
  map_size: 50,
  starting_ship: 604,
  starting_ship_maxed: true,
  friction_ratio: 1,
  shield_regen_factor: 20,
  power_regen_factor: 20,
  crystal_value: 5,
  asteroids_strength: 0,
  speed_mod: 2,
  release_crystal: true,
  survival_time: 40,
  
};

this.tick = function(game) {
  for (var ship of game.ships) {
    ship.set({
      crystals: 0,
    });
  }
}