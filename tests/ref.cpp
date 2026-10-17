// Prints per-pair estimate/boost and ghost landing positions for a fixed set of rays.
#include "lens.h"
#include "trace.h"
#include "ghost.h"
#include <cstdio>
#include <cmath>
int main(int argc, char** argv){
  LensSystem lens; if(!lens.load(argv[1])) return 1;
  float tanH = std::tan(40.f*3.14159265f/360.f), tanV = tanH/(1920.f/1080.f);
  float shw = lens.focal_length*tanH, shh = lens.focal_length*tanV;
  GhostConfig cfg; std::vector<GhostPair> pairs; std::vector<float> boosts;
  filter_ghost_pairs(lens, shw, shh, cfg, pairs, boosts);
  printf("%zu\n", pairs.size());
  float R0 = lens.surfaces[0].semi_aperture, z0 = lens.surfaces[0].z - 20.f;
  for(size_t i=0;i<pairs.size();++i){
    printf("P %d %d %.6g\n", pairs[i].surf_a, pairs[i].surf_b, boosts[i]);
    for(int k=0;k<5;k++){
      float u = -0.8f + 0.4f*k, v = 0.3f - 0.15f*k;
      Ray r; r.origin = Vec3f(u*R0, v*R0, z0); r.dir = Vec3f(std::tan(0.12f), std::tan(-0.08f), 1.f).normalized();
      TraceResult t = trace_ghost_ray(r, lens, pairs[i].surf_a, pairs[i].surf_b, 550.f);
      if(t.valid) printf("R %d %.6f %.6f %.6g\n", k, t.position.x, t.position.y, t.weight); else printf("R %d x\n", k);
    }
  }
}
