import Database from 'better-sqlite3';
import { join } from 'path';

const db = new Database(join(process.cwd(), 'data', 'quest-log.db'));

// DevOps Tools and Graphics Projects
const paths = [
  {
    name: 'Build Your Own Terraform',
    description: 'Create an infrastructure-as-code tool with state management and provider plugins',
    icon: 'cloud',
    color: 'purple',
    language: 'Go',
    skills: 'IaC, State management, Plugin architecture, Cloud APIs',
    difficulty: 'advanced',
    estimated_weeks: 8,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | HCL parsing | Config syntax |
| 1 | 2 | Resource blocks | Resource parsing |
| 1 | 3 | Variables | Input variables |
| 1 | 4 | Outputs | Output values |
| 1 | 5 | Locals | Local values |
| 2 | 1 | State file | State structure |
| 2 | 2 | State storage | Local backend |
| 2 | 3 | State locking | Lock mechanism |
| 2 | 4 | Remote state | S3 backend |
| 2 | 5 | State migration | Import/export |
| 3 | 1 | Provider interface | Plugin API |
| 3 | 2 | Resource CRUD | Create/Read/Update/Delete |
| 3 | 3 | Provider config | Configuration |
| 3 | 4 | AWS provider | EC2 resources |
| 3 | 5 | Data sources | Read-only resources |
| 4 | 1 | Dependency graph | Resource deps |
| 4 | 2 | Graph walking | Topological sort |
| 4 | 3 | Parallel execution | Concurrent ops |
| 4 | 4 | Error handling | Partial applies |
| 4 | 5 | Refresh | State refresh |
| 5 | 1 | Plan generation | Diff calculation |
| 5 | 2 | Plan display | Format output |
| 5 | 3 | Plan serialization | Save plans |
| 5 | 4 | Apply from plan | Execute plan |
| 5 | 5 | Destroy | Resource cleanup |
| 6 | 1 | Modules | Module blocks |
| 6 | 2 | Module sources | Local/remote |
| 6 | 3 | Module inputs | Variable passing |
| 6 | 4 | Module outputs | Output refs |
| 6 | 5 | Module registry | Public modules |
| 7 | 1 | Provisioners | Local exec |
| 7 | 2 | Remote exec | SSH execution |
| 7 | 3 | File provisioner | Copy files |
| 7 | 4 | Null resource | Meta-resource |
| 7 | 5 | Lifecycle | Hooks |
| 8 | 1 | Workspaces | Environments |
| 8 | 2 | CLI interface | Commands |
| 8 | 3 | Validation | Config validation |
| 8 | 4 | Format | Code formatting |
| 8 | 5 | Integration | Full tool |`,
    modules: [
      {
        name: 'Core Infrastructure',
        description: 'IaC core implementation',
        tasks: [
          {
            title: 'Terraform Clone in Go',
            description: 'Build infrastructure-as-code tool',
            details: `# Terraform-like IaC Tool in Go

\`\`\`go
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// Resource represents a managed resource
type Resource struct {
	Type       string                 \`json:"type"\`
	Name       string                 \`json:"name"\`
	Provider   string                 \`json:"provider"\`
	Config     map[string]interface{} \`json:"config"\`
	State      map[string]interface{} \`json:"state"\`
	DependsOn  []string               \`json:"depends_on"\`
}

// State represents the infrastructure state
type State struct {
	Version   int                  \`json:"version"\`
	Resources map[string]*Resource \`json:"resources"\`
	Outputs   map[string]interface{} \`json:"outputs"\`
	mu        sync.RWMutex
}

func NewState() *State {
	return &State{
		Version:   1,
		Resources: make(map[string]*Resource),
		Outputs:   make(map[string]interface{}),
	}
}

func (s *State) Save(path string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func (s *State) Load(path string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	return json.Unmarshal(data, s)
}

// Provider interface
type Provider interface {
	Configure(config map[string]interface{}) error
	CreateResource(resourceType string, config map[string]interface{}) (map[string]interface{}, error)
	ReadResource(resourceType string, id string) (map[string]interface{}, error)
	UpdateResource(resourceType string, id string, config map[string]interface{}) (map[string]interface{}, error)
	DeleteResource(resourceType string, id string) error
}

// AWS Provider (simplified)
type AWSProvider struct {
	Region    string
	AccessKey string
	SecretKey string
}

func (p *AWSProvider) Configure(config map[string]interface{}) error {
	if region, ok := config["region"].(string); ok {
		p.Region = region
	}
	return nil
}

func (p *AWSProvider) CreateResource(resourceType string, config map[string]interface{}) (map[string]interface{}, error) {
	switch resourceType {
	case "aws_instance":
		// Simulate EC2 instance creation
		return map[string]interface{}{
			"id":         fmt.Sprintf("i-%s", generateID()),
			"ami":        config["ami"],
			"instance_type": config["instance_type"],
			"public_ip":  "1.2.3.4",
			"private_ip": "10.0.0.1",
		}, nil
	case "aws_s3_bucket":
		return map[string]interface{}{
			"id":     config["bucket"],
			"bucket": config["bucket"],
			"arn":    fmt.Sprintf("arn:aws:s3:::%s", config["bucket"]),
		}, nil
	default:
		return nil, fmt.Errorf("unknown resource type: %s", resourceType)
	}
}

func (p *AWSProvider) ReadResource(resourceType string, id string) (map[string]interface{}, error) {
	// In real impl: call AWS API
	return nil, nil
}

func (p *AWSProvider) UpdateResource(resourceType string, id string, config map[string]interface{}) (map[string]interface{}, error) {
	// In real impl: call AWS API
	return config, nil
}

func (p *AWSProvider) DeleteResource(resourceType string, id string) error {
	fmt.Printf("Deleting %s: %s\\n", resourceType, id)
	return nil
}

// Configuration parser (simplified HCL-like)
type Config struct {
	Providers map[string]map[string]interface{} \`json:"providers"\`
	Resources []*ResourceConfig                 \`json:"resources"\`
	Variables map[string]interface{}            \`json:"variables"\`
	Outputs   map[string]string                 \`json:"outputs"\`
}

type ResourceConfig struct {
	Type      string                 \`json:"type"\`
	Name      string                 \`json:"name"\`
	Config    map[string]interface{} \`json:"config"\`
	DependsOn []string               \`json:"depends_on"\`
}

func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	return &config, nil
}

// Planner generates execution plans
type Plan struct {
	Creates []*Resource
	Updates []*Resource
	Deletes []*Resource
}

type Planner struct {
	config    *Config
	state     *State
	providers map[string]Provider
}

func NewPlanner(config *Config, state *State) *Planner {
	return &Planner{
		config:    config,
		state:     state,
		providers: make(map[string]Provider),
	}
}

func (p *Planner) RegisterProvider(name string, provider Provider) {
	p.providers[name] = provider
}

func (p *Planner) Plan() (*Plan, error) {
	plan := &Plan{}

	// Build desired state from config
	desired := make(map[string]*ResourceConfig)
	for _, rc := range p.config.Resources {
		key := fmt.Sprintf("%s.%s", rc.Type, rc.Name)
		desired[key] = rc
	}

	// Find creates and updates
	for key, rc := range desired {
		existing := p.state.Resources[key]
		if existing == nil {
			plan.Creates = append(plan.Creates, &Resource{
				Type:      rc.Type,
				Name:      rc.Name,
				Provider:  getProviderName(rc.Type),
				Config:    rc.Config,
				DependsOn: rc.DependsOn,
			})
		} else if !configsEqual(existing.Config, rc.Config) {
			plan.Updates = append(plan.Updates, &Resource{
				Type:      rc.Type,
				Name:      rc.Name,
				Provider:  getProviderName(rc.Type),
				Config:    rc.Config,
				State:     existing.State,
				DependsOn: rc.DependsOn,
			})
		}
	}

	// Find deletes
	for key, existing := range p.state.Resources {
		if _, ok := desired[key]; !ok {
			plan.Deletes = append(plan.Deletes, existing)
		}
	}

	return plan, nil
}

func (p *Planner) Apply(plan *Plan) error {
	// Topological sort based on dependencies
	// For simplicity, just process in order

	// Creates
	for _, r := range plan.Creates {
		fmt.Printf("Creating %s.%s...\\n", r.Type, r.Name)

		provider := p.providers[r.Provider]
		if provider == nil {
			return fmt.Errorf("provider not found: %s", r.Provider)
		}

		state, err := provider.CreateResource(r.Type, r.Config)
		if err != nil {
			return fmt.Errorf("create failed: %w", err)
		}

		r.State = state
		key := fmt.Sprintf("%s.%s", r.Type, r.Name)
		p.state.Resources[key] = r

		fmt.Printf("Created %s.%s: %v\\n", r.Type, r.Name, state["id"])
	}

	// Updates
	for _, r := range plan.Updates {
		fmt.Printf("Updating %s.%s...\\n", r.Type, r.Name)

		provider := p.providers[r.Provider]
		id := r.State["id"].(string)

		state, err := provider.UpdateResource(r.Type, id, r.Config)
		if err != nil {
			return fmt.Errorf("update failed: %w", err)
		}

		r.State = state
		key := fmt.Sprintf("%s.%s", r.Type, r.Name)
		p.state.Resources[key] = r

		fmt.Printf("Updated %s.%s\\n", r.Type, r.Name)
	}

	// Deletes (reverse order)
	for i := len(plan.Deletes) - 1; i >= 0; i-- {
		r := plan.Deletes[i]
		fmt.Printf("Destroying %s.%s...\\n", r.Type, r.Name)

		provider := p.providers[r.Provider]
		id := r.State["id"].(string)

		if err := provider.DeleteResource(r.Type, id); err != nil {
			return fmt.Errorf("delete failed: %w", err)
		}

		key := fmt.Sprintf("%s.%s", r.Type, r.Name)
		delete(p.state.Resources, key)

		fmt.Printf("Destroyed %s.%s\\n", r.Type, r.Name)
	}

	return nil
}

func getProviderName(resourceType string) string {
	// Extract provider from resource type (e.g., aws_instance -> aws)
	for i, c := range resourceType {
		if c == '_' {
			return resourceType[:i]
		}
	}
	return resourceType
}

func configsEqual(a, b map[string]interface{}) bool {
	aj, _ := json.Marshal(a)
	bj, _ := json.Marshal(b)
	return string(aj) == string(bj)
}

func generateID() string {
	return fmt.Sprintf("%08x", time.Now().UnixNano()&0xFFFFFFFF)
}

// CLI
func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: myterraform <plan|apply|destroy>")
		os.Exit(1)
	}

	// Load config
	config, err := LoadConfig("main.tf.json")
	if err != nil {
		fmt.Printf("Error loading config: %v\\n", err)
		os.Exit(1)
	}

	// Load state
	state := NewState()
	state.Load("terraform.tfstate")

	// Setup planner
	planner := NewPlanner(config, state)

	// Register providers
	aws := &AWSProvider{}
	aws.Configure(config.Providers["aws"])
	planner.RegisterProvider("aws", aws)

	switch os.Args[1] {
	case "plan":
		plan, err := planner.Plan()
		if err != nil {
			fmt.Printf("Plan error: %v\\n", err)
			os.Exit(1)
		}

		fmt.Println("\\nExecution Plan:")
		for _, r := range plan.Creates {
			fmt.Printf("  + %s.%s\\n", r.Type, r.Name)
		}
		for _, r := range plan.Updates {
			fmt.Printf("  ~ %s.%s\\n", r.Type, r.Name)
		}
		for _, r := range plan.Deletes {
			fmt.Printf("  - %s.%s\\n", r.Type, r.Name)
		}

	case "apply":
		plan, _ := planner.Plan()
		if err := planner.Apply(plan); err != nil {
			fmt.Printf("Apply error: %v\\n", err)
			os.Exit(1)
		}
		state.Save("terraform.tfstate")
		fmt.Println("\\nApply complete!")

	case "destroy":
		// Set config to empty to delete all
		config.Resources = nil
		planner = NewPlanner(config, state)
		planner.RegisterProvider("aws", aws)

		plan, _ := planner.Plan()
		if err := planner.Apply(plan); err != nil {
			fmt.Printf("Destroy error: %v\\n", err)
			os.Exit(1)
		}
		state.Save("terraform.tfstate")
		fmt.Println("\\nDestroy complete!")
	}
}
\`\`\``
          }
        ]
      }
    ]
  },
  {
    name: 'Build Your Own Ray Tracer',
    description: 'Create a physically-based ray tracer with materials, lighting, and acceleration structures',
    icon: 'sun',
    color: 'yellow',
    language: 'C++, Rust, C',
    skills: 'Computer graphics, Linear algebra, Physics simulation, Optimization',
    difficulty: 'advanced',
    estimated_weeks: 6,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | Vector math | Vec3 class |
| 1 | 2 | Ray class | Origin + direction |
| 1 | 3 | Sphere intersection | Ray-sphere |
| 1 | 4 | Image output | PPM format |
| 1 | 5 | Camera | View rays |
| 2 | 1 | Surface normals | Normal calculation |
| 2 | 2 | Multiple objects | Object list |
| 2 | 3 | Antialiasing | Multisampling |
| 2 | 4 | Diffuse materials | Lambertian |
| 2 | 5 | Metal materials | Reflection |
| 3 | 1 | Dielectrics | Glass/water |
| 3 | 2 | Refraction | Snell's law |
| 3 | 3 | Fresnel | Schlick approx |
| 3 | 4 | Positionable camera | FOV, look-at |
| 3 | 5 | Defocus blur | Depth of field |
| 4 | 1 | Textures | UV mapping |
| 4 | 2 | Perlin noise | Procedural tex |
| 4 | 3 | Image textures | Texture loading |
| 4 | 4 | Emissive materials | Light sources |
| 4 | 5 | Background | Environment maps |
| 5 | 1 | BVH basics | Bounding boxes |
| 5 | 2 | BVH construction | Tree building |
| 5 | 3 | BVH traversal | Fast intersection |
| 5 | 4 | Triangles | Triangle mesh |
| 5 | 5 | OBJ loading | Mesh import |
| 6 | 1 | Motion blur | Moving objects |
| 6 | 2 | Importance sampling | Better convergence |
| 6 | 3 | PDF | Light sampling |
| 6 | 4 | Multithreading | Parallel rendering |
| 6 | 5 | Final render | Complete scene |`,
    modules: [
      {
        name: 'Ray Tracing Core',
        description: 'Path tracing implementation',
        tasks: [
          {
            title: 'Ray Tracer in Rust',
            description: 'Build a path tracer with BVH acceleration',
            details: `# Ray Tracer in Rust

\`\`\`rust
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use rand::Rng;

// Vector3
#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    fn zero() -> Self {
        Vec3::new(0.0, 0.0, 0.0)
    }

    fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    fn normalize(&self) -> Self {
        let len = self.length();
        Vec3::new(self.x / len, self.y / len, self.z / len)
    }

    fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn reflect(&self, normal: &Vec3) -> Vec3 {
        *self - *normal * 2.0 * self.dot(normal)
    }

    fn refract(&self, normal: &Vec3, etai_over_etat: f64) -> Vec3 {
        let cos_theta = (-*self).dot(normal).min(1.0);
        let r_out_perp = (*self + *normal * cos_theta) * etai_over_etat;
        let r_out_parallel = *normal * -(1.0 - r_out_perp.length_squared()).abs().sqrt();
        r_out_perp + r_out_parallel
    }

    fn near_zero(&self) -> bool {
        let s = 1e-8;
        self.x.abs() < s && self.y.abs() < s && self.z.abs() < s
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, t: f64) -> Vec3 {
        Vec3::new(self.x * t, self.y * t, self.z * t)
    }
}

impl std::ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

type Color = Vec3;
type Point3 = Vec3;

// Ray
struct Ray {
    origin: Point3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: Point3, direction: Vec3) -> Self {
        Ray { origin, direction }
    }

    fn at(&self, t: f64) -> Point3 {
        self.origin + self.direction * t
    }
}

// Hit record
struct HitRecord {
    point: Point3,
    normal: Vec3,
    t: f64,
    front_face: bool,
    material: Arc<dyn Material>,
}

impl HitRecord {
    fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3) {
        self.front_face = ray.direction.dot(&outward_normal) < 0.0;
        self.normal = if self.front_face { outward_normal } else { -outward_normal };
    }
}

// Hittable trait
trait Hittable: Send + Sync {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
}

// Sphere
struct Sphere {
    center: Point3,
    radius: f64,
    material: Arc<dyn Material>,
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.length_squared();
        let half_b = oc.dot(&ray.direction);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();
        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || root > t_max {
            root = (-half_b + sqrtd) / a;
            if root < t_min || root > t_max {
                return None;
            }
        }

        let point = ray.at(root);
        let outward_normal = (point - self.center) * (1.0 / self.radius);

        let mut rec = HitRecord {
            point,
            normal: Vec3::zero(),
            t: root,
            front_face: false,
            material: Arc::clone(&self.material),
        };
        rec.set_face_normal(ray, outward_normal);
        Some(rec)
    }
}

// Material trait
trait Material: Send + Sync {
    fn scatter(&self, ray: &Ray, rec: &HitRecord) -> Option<(Color, Ray)>;
}

// Lambertian (diffuse)
struct Lambertian {
    albedo: Color,
}

impl Material for Lambertian {
    fn scatter(&self, _ray: &Ray, rec: &HitRecord) -> Option<(Color, Ray)> {
        let mut scatter_direction = rec.normal + random_unit_vector();
        if scatter_direction.near_zero() {
            scatter_direction = rec.normal;
        }
        let scattered = Ray::new(rec.point, scatter_direction);
        Some((self.albedo, scattered))
    }
}

// Metal
struct Metal {
    albedo: Color,
    fuzz: f64,
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, rec: &HitRecord) -> Option<(Color, Ray)> {
        let reflected = ray.direction.normalize().reflect(&rec.normal);
        let scattered = Ray::new(
            rec.point,
            reflected + random_in_unit_sphere() * self.fuzz,
        );
        if scattered.direction.dot(&rec.normal) > 0.0 {
            Some((self.albedo, scattered))
        } else {
            None
        }
    }
}

// Dielectric (glass)
struct Dielectric {
    ir: f64, // Index of refraction
}

impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, rec: &HitRecord) -> Option<(Color, Ray)> {
        let refraction_ratio = if rec.front_face { 1.0 / self.ir } else { self.ir };
        let unit_direction = ray.direction.normalize();

        let cos_theta = (-unit_direction).dot(&rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract = refraction_ratio * sin_theta > 1.0;

        let direction = if cannot_refract || reflectance(cos_theta, refraction_ratio) > rand::random() {
            unit_direction.reflect(&rec.normal)
        } else {
            unit_direction.refract(&rec.normal, refraction_ratio)
        };

        let scattered = Ray::new(rec.point, direction);
        Some((Color::new(1.0, 1.0, 1.0), scattered))
    }
}

fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
    // Schlick's approximation
    let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

// World (list of objects)
struct World {
    objects: Vec<Box<dyn Hittable>>,
}

impl Hittable for World {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut closest = t_max;
        let mut hit_record = None;

        for object in &self.objects {
            if let Some(rec) = object.hit(ray, t_min, closest) {
                closest = rec.t;
                hit_record = Some(rec);
            }
        }

        hit_record
    }
}

// Camera
struct Camera {
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f64,
}

impl Camera {
    fn new(
        lookfrom: Point3,
        lookat: Point3,
        vup: Vec3,
        vfov: f64,
        aspect_ratio: f64,
        aperture: f64,
        focus_dist: f64,
    ) -> Self {
        let theta = vfov * PI / 180.0;
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (lookfrom - lookat).normalize();
        let u = vup.cross(&w).normalize();
        let v = w.cross(&u);

        let origin = lookfrom;
        let horizontal = u * viewport_width * focus_dist;
        let vertical = v * viewport_height * focus_dist;
        let lower_left_corner = origin - horizontal * 0.5 - vertical * 0.5 - w * focus_dist;

        Camera {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
            u,
            v,
            lens_radius: aperture / 2.0,
        }
    }

    fn get_ray(&self, s: f64, t: f64) -> Ray {
        let rd = random_in_unit_disk() * self.lens_radius;
        let offset = self.u * rd.x + self.v * rd.y;

        Ray::new(
            self.origin + offset,
            self.lower_left_corner + self.horizontal * s + self.vertical * t - self.origin - offset,
        )
    }
}

// Random helpers
fn random_in_unit_sphere() -> Vec3 {
    let mut rng = rand::thread_rng();
    loop {
        let p = Vec3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        if p.length_squared() < 1.0 {
            return p;
        }
    }
}

fn random_unit_vector() -> Vec3 {
    random_in_unit_sphere().normalize()
}

fn random_in_unit_disk() -> Vec3 {
    let mut rng = rand::thread_rng();
    loop {
        let p = Vec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0);
        if p.length_squared() < 1.0 {
            return p;
        }
    }
}

// Ray color
fn ray_color(ray: &Ray, world: &World, depth: i32) -> Color {
    if depth <= 0 {
        return Color::zero();
    }

    if let Some(rec) = world.hit(ray, 0.001, f64::INFINITY) {
        if let Some((attenuation, scattered)) = rec.material.scatter(ray, &rec) {
            return attenuation * ray_color(&scattered, world, depth - 1);
        }
        return Color::zero();
    }

    // Background (sky gradient)
    let unit_direction = ray.direction.normalize();
    let t = 0.5 * (unit_direction.y + 1.0);
    Color::new(1.0, 1.0, 1.0) * (1.0 - t) + Color::new(0.5, 0.7, 1.0) * t
}

fn main() {
    // Image
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 400;
    let image_height = (image_width as f64 / aspect_ratio) as i32;
    let samples_per_pixel = 100;
    let max_depth = 50;

    // World
    let mut world = World { objects: Vec::new() };

    let mat_ground = Arc::new(Lambertian { albedo: Color::new(0.8, 0.8, 0.0) });
    let mat_center = Arc::new(Lambertian { albedo: Color::new(0.1, 0.2, 0.5) });
    let mat_left = Arc::new(Dielectric { ir: 1.5 });
    let mat_right = Arc::new(Metal { albedo: Color::new(0.8, 0.6, 0.2), fuzz: 0.0 });

    world.objects.push(Box::new(Sphere {
        center: Point3::new(0.0, -100.5, -1.0),
        radius: 100.0,
        material: mat_ground,
    }));
    world.objects.push(Box::new(Sphere {
        center: Point3::new(0.0, 0.0, -1.0),
        radius: 0.5,
        material: mat_center,
    }));
    world.objects.push(Box::new(Sphere {
        center: Point3::new(-1.0, 0.0, -1.0),
        radius: 0.5,
        material: mat_left,
    }));
    world.objects.push(Box::new(Sphere {
        center: Point3::new(1.0, 0.0, -1.0),
        radius: 0.5,
        material: mat_right,
    }));

    // Camera
    let camera = Camera::new(
        Point3::new(-2.0, 2.0, 1.0),
        Point3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 1.0, 0.0),
        20.0,
        aspect_ratio,
        0.1,
        3.4,
    );

    // Render
    let mut file = File::create("output.ppm").unwrap();
    write!(file, "P3\\n{} {}\\n255\\n", image_width, image_height).unwrap();

    for j in (0..image_height).rev() {
        eprint!("\\rScanlines remaining: {:3}", j);
        for i in 0..image_width {
            let mut pixel_color = Color::zero();

            for _ in 0..samples_per_pixel {
                let u = (i as f64 + rand::random::<f64>()) / (image_width - 1) as f64;
                let v = (j as f64 + rand::random::<f64>()) / (image_height - 1) as f64;
                let ray = camera.get_ray(u, v);
                pixel_color = pixel_color + ray_color(&ray, &world, max_depth);
            }

            // Write color
            let scale = 1.0 / samples_per_pixel as f64;
            let r = (256.0 * (pixel_color.x * scale).sqrt().clamp(0.0, 0.999)) as i32;
            let g = (256.0 * (pixel_color.y * scale).sqrt().clamp(0.0, 0.999)) as i32;
            let b = (256.0 * (pixel_color.z * scale).sqrt().clamp(0.0, 0.999)) as i32;

            writeln!(file, "{} {} {}", r, g, b).unwrap();
        }
    }

    eprintln!("\\nDone!");
}
\`\`\``
          }
        ]
      }
    ]
  },
  {
    name: 'Build Your Own TLS',
    description: 'Implement TLS 1.3 with handshake, key exchange, and encryption',
    icon: 'lock',
    color: 'green',
    language: 'C, Rust',
    skills: 'Cryptography, Network protocols, X.509, Key exchange',
    difficulty: 'advanced',
    estimated_weeks: 8,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | TLS overview | Protocol structure |
| 1 | 2 | Record layer | Record format |
| 1 | 3 | Handshake messages | Message types |
| 1 | 4 | Alert protocol | Error handling |
| 1 | 5 | TLS 1.3 changes | 1-RTT handshake |
| 2 | 1 | ClientHello | Hello construction |
| 2 | 2 | Extensions | SNI, ALPN |
| 2 | 3 | Supported versions | Version negotiation |
| 2 | 4 | Key share | ECDHE groups |
| 2 | 5 | ServerHello | Server response |
| 3 | 1 | X25519 | ECDH implementation |
| 3 | 2 | P-256 | NIST curve |
| 3 | 3 | Key derivation | HKDF |
| 3 | 4 | Traffic secrets | Secret schedule |
| 3 | 5 | Finished keys | Key confirmation |
| 4 | 1 | AES-GCM | AEAD encryption |
| 4 | 2 | ChaCha20-Poly1305 | Alt cipher |
| 4 | 3 | Nonce handling | IV/counter |
| 4 | 4 | Record encryption | Encrypt records |
| 4 | 5 | Record decryption | Decrypt records |
| 5 | 1 | Certificate parsing | X.509 basics |
| 5 | 2 | Chain validation | Trust chain |
| 5 | 3 | Certificate verify | Signature check |
| 5 | 4 | Finished message | MAC verification |
| 5 | 5 | Session tickets | Resumption |
| 6 | 1 | 0-RTT | Early data |
| 6 | 2 | PSK modes | Pre-shared keys |
| 6 | 3 | Certificate request | Client auth |
| 6 | 4 | Post-handshake | Key update |
| 6 | 5 | Close notify | Connection close |
| 7 | 1 | Socket integration | TCP + TLS |
| 7 | 2 | Non-blocking | Async I/O |
| 7 | 3 | Error handling | Alert generation |
| 7 | 4 | Buffering | Record assembly |
| 7 | 5 | State machine | Handshake states |
| 8 | 1 | Testing | Test vectors |
| 8 | 2 | Interop | OpenSSL test |
| 8 | 3 | Performance | Optimization |
| 8 | 4 | Security review | Timing attacks |
| 8 | 5 | Integration | Complete TLS |`,
    modules: [
      {
        name: 'TLS Implementation',
        description: 'TLS 1.3 protocol implementation',
        tasks: [
          {
            title: 'TLS 1.3 Handshake',
            description: 'Implement TLS 1.3 handshake protocol',
            details: `# TLS 1.3 Implementation Overview

TLS 1.3 implementation requires several cryptographic primitives and protocol state machine handling. Here's the core structure:

## Key Components

1. **Record Layer**: Fragments and encrypts/decrypts data
2. **Handshake Protocol**: Key exchange and authentication
3. **Alert Protocol**: Error signaling
4. **Application Data**: Encrypted user data

## TLS 1.3 Handshake Flow

\`\`\`
Client                                           Server

ClientHello
  + key_share
  + signature_algorithms
  + supported_versions  -------->
                                            ServerHello
                                              + key_share
                                              + supported_versions
                                  {EncryptedExtensions}
                                  {CertificateRequest*}
                                         {Certificate*}
                                   {CertificateVerify*}
                        <--------           {Finished}
{Certificate*}
{CertificateVerify*}
{Finished}              -------->
[Application Data]      <------->    [Application Data]
\`\`\`

## Implementation Notes

- Use X25519 for key exchange (faster than P-256)
- AES-128-GCM or ChaCha20-Poly1305 for encryption
- HKDF for key derivation
- Ed25519 or ECDSA for signatures

See the full implementation in the codebase for complete crypto operations.`
          }
        ]
      }
    ]
  },
  {
    name: 'Build Your Own Password Manager',
    description: 'Create a secure password manager with encryption, key derivation, and sync',
    icon: 'key',
    color: 'red',
    language: 'Rust, Go, C#',
    skills: 'Cryptography, Secure storage, Key derivation, UI development',
    difficulty: 'intermediate',
    estimated_weeks: 4,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | Architecture | Security design |
| 1 | 2 | Master password | Password input |
| 1 | 3 | Key derivation | Argon2id |
| 1 | 4 | Encryption key | Key stretching |
| 1 | 5 | Salt handling | Random salt |
| 2 | 1 | Vault format | JSON structure |
| 2 | 2 | Entry encryption | AES-256-GCM |
| 2 | 3 | Vault encryption | Full vault |
| 2 | 4 | File storage | Local save |
| 2 | 5 | Vault loading | Decrypt/parse |
| 3 | 1 | Password generation | Random passwords |
| 3 | 2 | Passphrase generation | Word lists |
| 3 | 3 | Entry management | CRUD operations |
| 3 | 4 | Search/filter | Find entries |
| 3 | 5 | Categories | Organization |
| 4 | 1 | Clipboard | Secure copy |
| 4 | 2 | Auto-clear | Memory wiping |
| 4 | 3 | Import/export | CSV/JSON |
| 4 | 4 | CLI interface | Commands |
| 4 | 5 | Security audit | Review |`,
    modules: [
      {
        name: 'Password Manager Core',
        description: 'Secure credential storage',
        tasks: [
          {
            title: 'Password Manager in Rust',
            description: 'Secure password vault with Argon2 and AES-GCM',
            details: `# Password Manager in Rust

\`\`\`rust
use aes_gcm::{Aes256Gcm, Key, Nonce, aead::{Aead, NewAead}};
use argon2::{Argon2, password_hash::SaltString};
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
use zeroize::Zeroize;

const SALT_LENGTH: usize = 16;
const NONCE_LENGTH: usize = 12;
const KEY_LENGTH: usize = 32;

#[derive(Serialize, Deserialize, Clone)]
pub struct Entry {
    pub id: String,
    pub name: String,
    pub username: String,
    pub password: String,
    pub url: Option<String>,
    pub notes: Option<String>,
    pub category: Option<String>,
    pub created_at: i64,
    pub modified_at: i64,
}

#[derive(Serialize, Deserialize)]
struct VaultData {
    entries: Vec<Entry>,
    version: u32,
}

#[derive(Serialize, Deserialize)]
struct EncryptedVault {
    salt: Vec<u8>,
    nonce: Vec<u8>,
    ciphertext: Vec<u8>,
    version: u32,
}

pub struct Vault {
    entries: Vec<Entry>,
    master_key: Vec<u8>,
    file_path: String,
}

impl Vault {
    pub fn create(master_password: &str, file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut salt = vec![0u8; SALT_LENGTH];
        OsRng.fill_bytes(&mut salt);

        let master_key = derive_key(master_password, &salt)?;

        let vault = Vault {
            entries: Vec::new(),
            master_key,
            file_path: file_path.to_string(),
        };

        vault.save()?;
        Ok(vault)
    }

    pub fn open(master_password: &str, file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = fs::read(file_path)?;
        let encrypted: EncryptedVault = bincode::deserialize(&data)?;

        let master_key = derive_key(master_password, &encrypted.salt)?;

        // Decrypt vault
        let cipher = Aes256Gcm::new(Key::from_slice(&master_key));
        let nonce = Nonce::from_slice(&encrypted.nonce);

        let plaintext = cipher.decrypt(nonce, encrypted.ciphertext.as_ref())
            .map_err(|_| "Decryption failed - wrong password?")?;

        let vault_data: VaultData = serde_json::from_slice(&plaintext)?;

        Ok(Vault {
            entries: vault_data.entries,
            master_key,
            file_path: file_path.to_string(),
        })
    }

    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let vault_data = VaultData {
            entries: self.entries.clone(),
            version: 1,
        };

        let plaintext = serde_json::to_vec(&vault_data)?;

        // Generate new nonce for each save
        let mut nonce_bytes = vec![0u8; NONCE_LENGTH];
        OsRng.fill_bytes(&mut nonce_bytes);

        // Generate new salt
        let mut salt = vec![0u8; SALT_LENGTH];
        OsRng.fill_bytes(&mut salt);

        // Encrypt
        let cipher = Aes256Gcm::new(Key::from_slice(&self.master_key));
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher.encrypt(nonce, plaintext.as_ref())
            .map_err(|_| "Encryption failed")?;

        let encrypted = EncryptedVault {
            salt,
            nonce: nonce_bytes,
            ciphertext,
            version: 1,
        };

        let data = bincode::serialize(&encrypted)?;
        fs::write(&self.file_path, data)?;

        Ok(())
    }

    pub fn add_entry(&mut self, entry: Entry) {
        self.entries.push(entry);
    }

    pub fn get_entry(&self, id: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn update_entry(&mut self, id: &str, entry: Entry) -> bool {
        if let Some(idx) = self.entries.iter().position(|e| e.id == id) {
            self.entries[idx] = entry;
            true
        } else {
            false
        }
    }

    pub fn delete_entry(&mut self, id: &str) -> bool {
        if let Some(idx) = self.entries.iter().position(|e| e.id == id) {
            self.entries.remove(idx);
            true
        } else {
            false
        }
    }

    pub fn search(&self, query: &str) -> Vec<&Entry> {
        let query = query.to_lowercase();
        self.entries.iter()
            .filter(|e| {
                e.name.to_lowercase().contains(&query) ||
                e.username.to_lowercase().contains(&query) ||
                e.url.as_ref().map(|u| u.to_lowercase().contains(&query)).unwrap_or(false)
            })
            .collect()
    }

    pub fn list_entries(&self) -> &[Entry] {
        &self.entries
    }
}

impl Drop for Vault {
    fn drop(&mut self) {
        self.master_key.zeroize();
    }
}

fn derive_key(password: &str, salt: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let argon2 = Argon2::new(
        argon2::Algorithm::Argon2id,
        argon2::Version::V0x13,
        argon2::Params::new(65536, 3, 4, Some(KEY_LENGTH))?,
    );

    let mut key = vec![0u8; KEY_LENGTH];
    argon2.hash_password_into(password.as_bytes(), salt, &mut key)?;

    Ok(key)
}

pub fn generate_password(length: usize, options: &PasswordOptions) -> String {
    let mut charset = String::new();

    if options.lowercase {
        charset.push_str("abcdefghijklmnopqrstuvwxyz");
    }
    if options.uppercase {
        charset.push_str("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    }
    if options.digits {
        charset.push_str("0123456789");
    }
    if options.symbols {
        charset.push_str("!@#\$%^&*()-_=+[]{}|;:,.<>?");
    }

    let charset: Vec<char> = charset.chars().collect();
    let mut password = String::with_capacity(length);

    for _ in 0..length {
        let idx = (OsRng.next_u32() as usize) % charset.len();
        password.push(charset[idx]);
    }

    password
}

pub struct PasswordOptions {
    pub lowercase: bool,
    pub uppercase: bool,
    pub digits: bool,
    pub symbols: bool,
}

impl Default for PasswordOptions {
    fn default() -> Self {
        PasswordOptions {
            lowercase: true,
            uppercase: true,
            digits: true,
            symbols: true,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("create") => {
            let password = rpassword::prompt_password("Master password: ")?;
            let _vault = Vault::create(&password, "vault.db")?;
            println!("Vault created!");
        }
        Some("add") => {
            let password = rpassword::prompt_password("Master password: ")?;
            let mut vault = Vault::open(&password, "vault.db")?;

            let entry = Entry {
                id: uuid::Uuid::new_v4().to_string(),
                name: args.get(2).cloned().unwrap_or_default(),
                username: args.get(3).cloned().unwrap_or_default(),
                password: generate_password(20, &PasswordOptions::default()),
                url: None,
                notes: None,
                category: None,
                created_at: chrono::Utc::now().timestamp(),
                modified_at: chrono::Utc::now().timestamp(),
            };

            println!("Generated password: {}", entry.password);
            vault.add_entry(entry);
            vault.save()?;
            println!("Entry added!");
        }
        Some("list") => {
            let password = rpassword::prompt_password("Master password: ")?;
            let vault = Vault::open(&password, "vault.db")?;

            for entry in vault.list_entries() {
                println!("{}: {} ({})", entry.name, entry.username, entry.id);
            }
        }
        Some("get") => {
            let password = rpassword::prompt_password("Master password: ")?;
            let vault = Vault::open(&password, "vault.db")?;

            if let Some(id) = args.get(2) {
                if let Some(entry) = vault.get_entry(id) {
                    println!("Name: {}", entry.name);
                    println!("Username: {}", entry.username);
                    println!("Password: {}", entry.password);
                }
            }
        }
        _ => {
            println!("Usage: pwmgr <create|add|list|get> [args]");
        }
    }

    Ok(())
}
\`\`\``
          }
        ]
      }
    ]
  }
];

// Insert all data
const insertPath = db.prepare(`
  INSERT INTO paths (name, description, icon, color, language, skills, difficulty, estimated_weeks, schedule)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
`);

const insertModule = db.prepare(`
  INSERT INTO modules (path_id, name, description)
  VALUES (?, ?, ?)
`);

const insertTask = db.prepare(`
  INSERT INTO tasks (module_id, title, description, details)
  VALUES (?, ?, ?, ?)
`);

for (const path of paths) {
  const pathResult = insertPath.run(
    path.name,
    path.description,
    path.icon,
    path.color,
    path.language,
    path.skills,
    path.difficulty,
    path.estimated_weeks,
    path.schedule
  );
  const pathId = pathResult.lastInsertRowid;

  for (const module of path.modules) {
    const moduleResult = insertModule.run(pathId, module.name, module.description);
    const moduleId = moduleResult.lastInsertRowid;

    for (const task of module.tasks) {
      insertTask.run(moduleId, task.title, task.description, task.details);
    }
  }
}

console.log('Seeded: Terraform, Ray Tracer, TLS, Password Manager');
